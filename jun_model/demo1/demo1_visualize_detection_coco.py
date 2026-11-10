import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COCO mitotic detection targets.")
    parser.add_argument(
        "--coco",
        default="jun_model/demo1/detection_coco/Aperio_CS2/test_coco.json",
        help="COCO JSON file to visualize.",
    )
    parser.add_argument("--image-index", type=int, default=0, help="Image index inside the COCO images list.")
    parser.add_argument("--max-size", type=int, default=1600, help="Maximum preview width or height.")
    parser.add_argument("--output", default="jun_model/demo1/outputs/detection_target_preview.jpg", help="Output preview JPG.")
    return parser.parse_args()


def main():
    args = parse_args()
    coco_path = Path(args.coco)
    coco = load_json(coco_path)
    image_info = coco["images"][args.image_index]
    image_id = image_info["id"]
    annotations = [annotation for annotation in coco["annotations"] if annotation["image_id"] == image_id]

    image_path = (coco_path.parent / image_info["file_name"]).resolve()
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(args.max_size / width, args.max_size / height, 1.0)
        preview = image.resize((int(width * scale), int(height * scale)))

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
        draw.rectangle(box, outline=(255, 40, 40), width=3)
        draw.text((box[0], max(0, box[1] - 12)), "mitotic", fill=(255, 40, 40), font=font)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output, quality=95)
    print(f"image: {image_path}")
    print(f"mitotic boxes: {len(annotations)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
