import json
from pathlib import Path


RESULTS = [
    ("Feature baseline", Path("demo1/models/feature_baseline/metrics.json")),
    ("Pixel baseline", Path("demo1/models/pixel_baseline/metrics.json")),
    ("Pixel + stain/color aug", Path("demo1/models/pixel_augmented_baseline/metrics.json")),
]


def fmt(value):
    return f"{value:.3f}"


def main():
    rows = []
    for name, path in RESULTS:
        with path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        train = metrics["train"]
        test = metrics["held_out_scanner_test"]
        rows.append(
            [
                name,
                fmt(train["f1"]),
                fmt(test["f1"]),
                fmt(test["precision"]),
                fmt(test["recall"]),
                fmt(test["accuracy"]),
            ]
        )

    header = ["Model", "Train F1", "Test F1", "Test Precision", "Test Recall", "Test Accuracy"]
    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]

    def line(row):
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row)))

    print(line(header))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


if __name__ == "__main__":
    main()
