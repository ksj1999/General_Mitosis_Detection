import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CATEGORY_COLORS = {
    1: (255, 60, 60),
    2: (50, 180, 255),
}


def load_subset(subset_dir):
    annotation_path = subset_dir / "metadata" / "annotations_breast_cancer.json"
    with annotation_path.open(encoding="utf-8") as f:
        data = json.load(f)

    images_by_name = {image["file_name"]: image for image in data["images"]}
    categories = {category["id"]: category["name"] for category in data["categories"]}
    annotations_by_image = {}

    for annotation in data["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    return images_by_name, annotations_by_image, categories


def draw_annotations(image_path, image_info, annotations, categories, output_path, max_size):
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size

        scale = min(max_size / original_width, max_size / original_height, 1.0)
        preview_size = (int(original_width * scale), int(original_height * scale))
        preview = image.resize(preview_size)

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()

    for annotation in annotations:
        x1, y1, x2, y2 = annotation["bbox"]
        category_id = annotation["category_id"]
        color = CATEGORY_COLORS.get(category_id, (255, 255, 0))

        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(box, outline=color, width=3)

        label = f"{category_id}: {categories.get(category_id, 'unknown')}"
        text_xy = (box[0], max(0, box[1] - 12))
        draw.text(text_xy, label, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, quality=95)

    return {
        "file_name": image_info["file_name"],
        "image_id": image_info["id"],
        "original_size": (original_width, original_height),
        "preview_size": preview_size,
        "annotation_count": len(annotations),
        "output_path": str(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a preview image with MIDOG++ breast cancer bounding boxes overlaid."
    )
    parser.add_argument("--subset-dir", default="jun_model/breast_cancer_subset", help="Path to the prepared subset folder.")
    parser.add_argument("--image", default="001.tiff", help="Image filename inside subset/images.")
    parser.add_argument("--max-size", type=int, default=1600, help="Maximum preview width or height in pixels.")
    parser.add_argument("--output-dir", default="jun_model/demo1/outputs", help="Directory for generated preview images.")
    return parser.parse_args()


def main():
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    image_path = subset_dir / "images" / args.image

    images_by_name, annotations_by_image, categories = load_subset(subset_dir)

    if args.image not in images_by_name:
        available = ", ".join(sorted(images_by_name)[:5])
        raise ValueError(f"{args.image} is not in the annotation JSON. First available files: {available}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_info = images_by_name[args.image]
    annotations = annotations_by_image.get(image_info["id"], [])
    output_path = Path(args.output_dir) / f"{Path(args.image).stem}_annotations.jpg"

    result = draw_annotations(
        image_path=image_path,
        image_info=image_info,
        annotations=annotations,
        categories=categories,
        output_path=output_path,
        max_size=args.max_size,
    )

    print("Created annotation preview")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
