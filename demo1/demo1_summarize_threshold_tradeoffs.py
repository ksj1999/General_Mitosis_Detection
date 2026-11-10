import argparse
import csv
from pathlib import Path


def read_metrics(path, label):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["model"] = label
        for key in ["score_threshold", "precision", "recall", "f1"]:
            row[key] = float(row[key])
        for key in ["tp", "fp", "fn"]:
            row[key] = int(row[key])
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize precision/recall threshold tradeoffs.")
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--comparison-metrics", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--comparison-name", default="comparison")
    parser.add_argument("--output-dir", default="demo1/outputs/threshold_tradeoffs_s360")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(args.baseline_metrics, args.baseline_name)
    rows.extend(read_metrics(args.comparison_metrics, args.comparison_name))

    with (output_dir / "threshold_tradeoffs.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "score_threshold", "precision", "recall", "f1", "tp", "fp", "fn"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])

    lines = [
        "# Threshold Tradeoff Summary",
        "",
        "| Model | Score | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['score_threshold']:.2f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )

    best_by_model = {}
    for row in rows:
        if row["model"] not in best_by_model or row["f1"] > best_by_model[row["model"]]["f1"]:
            best_by_model[row["model"]] = row

    lines.extend(["", "## Best F1", ""])
    for model, row in best_by_model.items():
        lines.append(
            f"- {model}: F1 {row['f1']:.3f} at score {row['score_threshold']:.2f} "
            f"(precision {row['precision']:.3f}, recall {row['recall']:.3f})"
        )

    (output_dir / "threshold_tradeoffs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
