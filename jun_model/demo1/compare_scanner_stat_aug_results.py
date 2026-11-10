import csv
from collections import defaultdict
from pathlib import Path


def fmt(value):
    return f"{float(value):.3f}"


def main():
    path = Path("jun_model/demo1/models/loso_scanner_stat_aug/results.csv")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_scanner = defaultdict(dict)
    by_model = defaultdict(list)
    for row in rows:
        by_scanner[row["held_out_scanner"]][row["model"]] = row
        by_model[row["model"]].append(row)

    header = ["Held-out scanner", "Baseline F1", "Generic Aug F1", "Scanner-Stat F1"]
    table = []
    for scanner in sorted(by_scanner):
        scanner_rows = by_scanner[scanner]
        table.append(
            [
                scanner,
                fmt(scanner_rows["baseline"]["test_f1"]),
                fmt(scanner_rows["generic_aug"]["test_f1"]),
                fmt(scanner_rows["scanner_stat_aug"]["test_f1"]),
            ]
        )

    mean_row = ["Mean"]
    for model in ["baseline", "generic_aug", "scanner_stat_aug"]:
        mean_f1 = sum(float(row["test_f1"]) for row in by_model[model]) / len(by_model[model])
        mean_row.append(fmt(mean_f1))
    table.append(mean_row)

    widths = [max(len(str(row[i])) for row in [header] + table) for i in range(len(header))]

    def line(row):
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row)))

    print(line(header))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(line(row))


if __name__ == "__main__":
    main()
