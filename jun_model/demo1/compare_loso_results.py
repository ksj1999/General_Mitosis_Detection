import csv
from pathlib import Path


def fmt(value):
    return f"{float(value):.3f}"


def main():
    path = Path("jun_model/demo1/models/loso_pixel/loso_results.csv")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_scanner = {}
    for row in rows:
        by_scanner.setdefault(row["held_out_scanner"], {})[row["model"]] = row

    table = []
    for scanner in sorted(by_scanner):
        baseline = by_scanner[scanner]["pixel_baseline"]
        augmented = by_scanner[scanner]["pixel_augmented"]
        baseline_f1 = float(baseline["test_f1"])
        augmented_f1 = float(augmented["test_f1"])
        table.append(
            [
                scanner,
                fmt(baseline_f1),
                fmt(augmented_f1),
                f"{augmented_f1 - baseline_f1:+.3f}",
                fmt(baseline["test_recall"]),
                fmt(augmented["test_recall"]),
            ]
        )

    header = ["Held-out scanner", "Pixel F1", "Aug F1", "Delta F1", "Pixel Recall", "Aug Recall"]
    widths = [max(len(str(row[i])) for row in [header] + table) for i in range(len(header))]

    def line(row):
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row)))

    print(line(header))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(line(row))


if __name__ == "__main__":
    main()
