import csv
from pathlib import Path


RUNS = [
    ("Faster R-CNN", Path("jun_model/demo1/models/detection_frcnn_aperio_b4/eval/metrics.csv")),
    ("Faster R-CNN + color jitter", Path("jun_model/demo1/models/detection_frcnn_aperio_color_jitter/eval/metrics.csv")),
]


def read_best(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return max(rows, key=lambda row: float(row["f1"]))


def fmt(value):
    return f"{float(value):.3f}"


def main():
    table = []
    for name, path in RUNS:
        best = read_best(path)
        table.append(
            [
                name,
                fmt(best["score_threshold"]),
                fmt(best["precision"]),
                fmt(best["recall"]),
                fmt(best["f1"]),
                best["tp"],
                best["fp"],
                best["fn"],
            ]
        )

    header = ["Model", "Score", "Precision", "Recall", "F1", "TP", "FP", "FN"]
    widths = [max(len(str(row[i])) for row in [header] + table) for i in range(len(header))]

    def line(row):
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row)))

    print(line(header))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(line(row))


if __name__ == "__main__":
    main()
