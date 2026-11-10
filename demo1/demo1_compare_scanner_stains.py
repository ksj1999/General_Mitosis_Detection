import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_summary(path, sample_size=512):
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((sample_size, sample_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    pixels = arr.reshape(-1, 3)
    return pixels.mean(axis=0), pixels.std(axis=0), image.copy()


def main():
    subset_dir = Path("breast_cancer_subset")
    rows = read_rows(subset_dir / "metadata" / "breast_cancer_slides.csv")
    by_scanner = defaultdict(list)
    for row in rows:
        by_scanner[row["Scanner"]].append(row)

    output_dir = Path("demo1/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    scanner_stats = {}
    thumbnails = []
    for scanner in sorted(by_scanner):
        means = []
        stds = []
        sample_rows = by_scanner[scanner][:6]
        for row in by_scanner[scanner]:
            mean, std, _ = image_summary(subset_dir / "images" / row["file_name"])
            means.append(mean)
            stds.append(std)

        scanner_stats[scanner] = {
            "mean_rgb": np.vstack(means).mean(axis=0),
            "std_rgb": np.vstack(stds).mean(axis=0),
        }

        for row in sample_rows:
            _, _, thumb = image_summary(subset_dir / "images" / row["file_name"], sample_size=220)
            thumbnails.append((scanner, row["file_name"], thumb.resize((220, 165))))

    tile_w, tile_h = 220, 190
    cols = 6
    scanners = sorted(by_scanner)
    sheet = Image.new("RGB", (cols * tile_w, len(scanners) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for scanner_index, scanner in enumerate(scanners):
        scanner_thumbs = [item for item in thumbnails if item[0] == scanner]
        for col, (_, file_name, thumb) in enumerate(scanner_thumbs):
            x = col * tile_w
            y = scanner_index * tile_h
            sheet.paste(thumb, (x, y + 20))
            draw.text((x + 4, y + 4), f"{scanner} / {file_name}", fill=(0, 0, 0), font=font)

    sheet.save(output_dir / "scanner_stain_comparison.jpg", quality=95)

    with (output_dir / "scanner_color_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scanner", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"])
        for scanner, stats in scanner_stats.items():
            writer.writerow(
                [
                    scanner,
                    *[f"{value:.4f}" for value in stats["mean_rgb"]],
                    *[f"{value:.4f}" for value in stats["std_rgb"]],
                ]
            )

    for scanner, stats in scanner_stats.items():
        mean = ", ".join(f"{x:.3f}" for x in stats["mean_rgb"])
        std = ", ".join(f"{x:.3f}" for x in stats["std_rgb"])
        print(f"{scanner}: mean RGB=({mean}), std RGB=({std})")

    print(f"preview: {output_dir / 'scanner_stain_comparison.jpg'}")
    print(f"stats: {output_dir / 'scanner_color_stats.csv'}")


if __name__ == "__main__":
    main()
