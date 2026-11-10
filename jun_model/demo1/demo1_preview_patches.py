import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args():
    parser = argparse.ArgumentParser(description="Create a contact-sheet preview of generated patch samples.")
    parser.add_argument("--patch-dir", default="jun_model/demo1/patch_dataset", help="Patch dataset directory.")
    parser.add_argument("--output", default="jun_model/demo1/outputs/patch_preview.jpg", help="Output preview image.")
    parser.add_argument("--samples-per-label", type=int, default=8, help="Samples per label to show.")
    return parser.parse_args()


def main():
    args = parse_args()
    patch_dir = Path(args.patch_dir)
    rows = read_rows(patch_dir / "patch_metadata.csv")

    selected = []
    for label in ["mitotic_figure", "not_mitotic_figure"]:
        label_rows = [row for row in rows if row["split"] == "train" and row["label"] == label]
        selected.extend(label_rows[: args.samples_per_label])

    tile_size = 128
    label_height = 18
    cols = args.samples_per_label
    rows_count = 2
    preview = Image.new("RGB", (cols * tile_size, rows_count * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for idx, row in enumerate(selected):
        grid_y = 0 if row["label"] == "mitotic_figure" else 1
        grid_x = idx % args.samples_per_label
        x = grid_x * tile_size
        y = grid_y * (tile_size + label_height)

        patch = Image.open(patch_dir / row["patch_path"]).convert("RGB")
        preview.paste(patch, (x, y))
        draw.text((x + 2, y + tile_size + 2), row["label"].replace("_", " "), fill=(0, 0, 0), font=font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output, quality=95)
    print(f"preview: {output}")


if __name__ == "__main__":
    main()
