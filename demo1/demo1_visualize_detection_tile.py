import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a tiled COCO detection sample.")
    parser.add_argument(
        "--coco",
        default="demo1/detection_tiles/Aperio_CS2/train_tiles_coco.json",
        help="Tile COCO JSON.",
    )
    parser.add_argument("--image-index", type=int, default=0, help="Tile image index.")
    parser.add_argument("--output", default="demo1/outputs/detection_tile_preview.jpg", help="Output image.")
    return parser.parse_args()


def main():
    args = parse_args()
    coco_path = Path(args.coco)
    coco = load_json(coco_path)
    image_info = coco["images"][args.image_index]
    image_id = image_info["id"]
    annotations = [annotation for annotation in coco["annotations"] if annotation["image_id"] == image_id]

    image_path = coco_path.parent / image_info["file_name"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=3)
        draw.text((x, max(0, y - 12)), "mitotic", fill=(255, 40, 40), font=font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)
    print(f"tile: {image_path}")
    print(f"boxes: {len(annotations)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
