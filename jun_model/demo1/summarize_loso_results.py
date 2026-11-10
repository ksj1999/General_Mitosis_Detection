import csv
from collections import defaultdict
from pathlib import Path


METRICS = ["test_f1", "test_precision", "test_recall", "test_accuracy"]


def fmt(value):
    return f"{float(value):.3f}"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values)


def markdown_table(header, rows):
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    input_path = Path("jun_model/demo1/models/loso_pixel/loso_results.csv")
    output_path = Path("jun_model/demo1/models/loso_pixel/loso_summary.md")
    rows = read_rows(input_path)

    by_model = defaultdict(list)
    by_scanner = defaultdict(dict)
    for row in rows:
        by_model[row["model"]].append(row)
        by_scanner[row["held_out_scanner"]][row["model"]] = row

    scanner_rows = []
    for scanner in sorted(by_scanner):
        baseline = by_scanner[scanner]["pixel_baseline"]
        augmented = by_scanner[scanner]["pixel_augmented"]
        scanner_rows.append(
            [
                scanner,
                fmt(baseline["test_f1"]),
                fmt(augmented["test_f1"]),
                f"{float(augmented['test_f1']) - float(baseline['test_f1']):+.3f}",
                fmt(baseline["test_recall"]),
                fmt(augmented["test_recall"]),
            ]
        )

    mean_rows = []
    for model in ["pixel_baseline", "pixel_augmented"]:
        model_rows = by_model[model]
        mean_rows.append(
            [
                model,
                *[fmt(mean(row[metric] for row in model_rows)) for metric in METRICS],
            ]
        )

    baseline_mean_f1 = mean(row["test_f1"] for row in by_model["pixel_baseline"])
    augmented_mean_f1 = mean(row["test_f1"] for row in by_model["pixel_augmented"])
    baseline_mean_recall = mean(row["test_recall"] for row in by_model["pixel_baseline"])
    augmented_mean_recall = mean(row["test_recall"] for row in by_model["pixel_augmented"])

    summary = [
        "# Demo 1 LOSO Summary",
        "",
        "Leave-one-scanner-out patch classification on the MIDOG++ human breast cancer subset.",
        "",
        "## Per-Scanner Results",
        "",
        markdown_table(
            ["Held-out scanner", "Pixel F1", "Aug F1", "Delta F1", "Pixel Recall", "Aug Recall"],
            scanner_rows,
        ),
        "",
        "## Mean Held-Out Performance",
        "",
        markdown_table(
            ["Model", "Mean F1", "Mean Precision", "Mean Recall", "Mean Accuracy"],
            mean_rows,
        ),
        "",
        "## Key Takeaway",
        "",
        (
            f"Stain/color augmentation changed mean held-out F1 from {baseline_mean_f1:.3f} "
            f"to {augmented_mean_f1:.3f} ({augmented_mean_f1 - baseline_mean_f1:+.3f}) "
            f"and mean held-out recall from {baseline_mean_recall:.3f} to "
            f"{augmented_mean_recall:.3f} ({augmented_mean_recall - baseline_mean_recall:+.3f})."
        ),
        "",
        "This supports the project direction, while still being a patch-level result rather than full-image detection.",
        "",
    ]

    output_path.write_text("\n".join(summary), encoding="utf-8")
    print(output_path)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
