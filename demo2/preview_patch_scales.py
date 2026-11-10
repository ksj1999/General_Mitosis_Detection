import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from patch_utils import annotation_box_in_patch, extract_candidate_patch


SCALES = ("20x", "40x")
OUTPUT_SIZE = 224


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_patch(image, row, scale):
    annotated = extract_candidate_patch(image, row, scale, OUTPUT_SIZE)
    x1, y1, x2, y2 = annotation_box_in_patch(row, scale, OUTPUT_SIZE)
    color = (30, 220, 80) if row["label"] == "1" else (255, 190, 30)
    ImageDraw.Draw(annotated).rectangle((x1, y1, x2, y2), outline=color, width=2)
    return annotated


def choose_examples(rows, per_group, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scanner"], row["label_name"])].append(row)
    rng = random.Random(seed)
    selected = []
    for key in sorted(grouped):
        preferred = [
            row
            for row in grouped[key]
            if row["touches_edge_20x"] == "False" and row["ambiguous_negative_20x"] == "False"
        ]
        pool = preferred or grouped[key]
        selected.extend(rng.sample(pool, min(per_group, len(pool))))
    return selected


def make_preview(rows, output_path, per_group, seed):
    selected = choose_examples(rows, per_group, seed)
    columns = 3
    cell_width = OUTPUT_SIZE * 2 + 18
    cell_height = OUTPUT_SIZE + 48
    grid_rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, grid_rows * cell_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    opened_file = None
    image = None
    try:
        for index, row in enumerate(selected):
            if opened_file != row["image_path"]:
                if image is not None:
                    image.close()
                image = Image.open(row["image_path"])
                opened_file = row["image_path"]
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            title = f"{row['scanner']} | {row['label_name']} | {row['file_name']} #{row['candidate_id']}"
            draw.text((x + 4, y + 4), title, fill=(20, 20, 20), font=font)
            for scale_index, scale in enumerate(SCALES):
                patch = extract_patch(image, row, scale)
                patch_x = x + 4 + scale_index * (OUTPUT_SIZE + 6)
                patch_y = y + 22
                sheet.paste(patch, (patch_x, patch_y))
                draw.text((patch_x + 4, patch_y + OUTPUT_SIZE + 3), scale, fill=(20, 20, 20), font=font)
    finally:
        if image is not None:
            image.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)
    return len(selected)


def parse_args():
    parser = argparse.ArgumentParser(description="Create side-by-side 20x and 40x candidate patch checks.")
    parser.add_argument("--manifest", type=Path, default=Path("demo2/data/candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path("demo2/outputs/patch_scale_preview.jpg"))
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    count = make_preview(read_rows(args.manifest), args.output, args.per_group, args.seed)
    print(f"preview examples: {count}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
