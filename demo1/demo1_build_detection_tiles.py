import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


TILE_SIZE = 512


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clamp_tile_origin(left, top, image_width, image_height, tile_size):
    left = max(0, min(int(round(left)), image_width - tile_size))
    top = max(0, min(int(round(top)), image_height - tile_size))
    return left, top


def bbox_center(annotation):
    x, y, w, h = annotation["bbox"]
    return x + w / 2, y + h / 2


def boxes_for_tile(annotations, left, top, tile_size):
    tile_annotations = []
    for annotation in annotations:
        center_x, center_y = bbox_center(annotation)
        if not (left <= center_x < left + tile_size and top <= center_y < top + tile_size):
            continue

        x, y, w, h = annotation["bbox"]
        clipped_x1 = max(x, left)
        clipped_y1 = max(y, top)
        clipped_x2 = min(x + w, left + tile_size)
        clipped_y2 = min(y + h, top + tile_size)
        clipped_w = clipped_x2 - clipped_x1
        clipped_h = clipped_y2 - clipped_y1

        if clipped_w <= 0 or clipped_h <= 0:
            continue

        tile_annotations.append(
            {
                "category_id": 1,
                "bbox": [
                    clipped_x1 - left,
                    clipped_y1 - top,
                    clipped_w,
                    clipped_h,
                ],
                "area": clipped_w * clipped_h,
                "iscrowd": 0,
                "source_annotation_id": annotation["id"],
            }
        )
    return tile_annotations


def tile_has_mitosis_center(annotations, left, top, tile_size):
    for annotation in annotations:
        center_x, center_y = bbox_center(annotation)
        if left <= center_x < left + tile_size and top <= center_y < top + tile_size:
            return True
    return False


def save_tile(source_image, left, top, tile_size, output_path):
    right = min(left + tile_size, source_image.width)
    bottom = min(top + tile_size, source_image.height)
    crop = source_image.crop((left, top, right, bottom))
    tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
    tile.paste(crop, (0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile.save(output_path, quality=95)


def build_split(coco_path, output_dir, split_name, tile_size, negative_ratio, seed):
    rng = random.Random(seed)
    coco = load_json(coco_path)
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    tile_images = []
    tile_annotations = []
    next_image_id = 1
    next_annotation_id = 1
    positive_tiles = 0
    negative_tiles = 0

    Image.MAX_IMAGE_PIXELS = None

    for image_info in coco["images"]:
        source_path = (coco_path.parent / image_info["file_name"]).resolve()
        source_annotations = annotations_by_image.get(image_info["id"], [])

        with Image.open(source_path) as source_image:
            source_image = source_image.convert("RGB")
            image_width, image_height = source_image.size

            positive_origins = []
            for annotation in source_annotations:
                center_x, center_y = bbox_center(annotation)
                left, top = clamp_tile_origin(
                    center_x - tile_size / 2,
                    center_y - tile_size / 2,
                    image_width,
                    image_height,
                    tile_size,
                )
                origin = (left, top)
                if origin not in positive_origins:
                    positive_origins.append(origin)

            for left, top in positive_origins:
                tile_anns = boxes_for_tile(source_annotations, left, top, tile_size)
                if not tile_anns:
                    continue

                tile_file = f"{split_name}/positive/{Path(image_info['file_name']).stem}_x{left}_y{top}.jpg"
                save_tile(source_image, left, top, tile_size, output_dir / tile_file)
                tile_images.append(
                    {
                        "id": next_image_id,
                        "file_name": tile_file.replace("\\", "/"),
                        "width": tile_size,
                        "height": tile_size,
                        "source_image_id": image_info["id"],
                        "source_file_name": image_info["file_name"],
                        "tile_left": left,
                        "tile_top": top,
                    }
                )
                for tile_ann in tile_anns:
                    tile_ann["id"] = next_annotation_id
                    tile_ann["image_id"] = next_image_id
                    tile_annotations.append(tile_ann)
                    next_annotation_id += 1
                next_image_id += 1
                positive_tiles += 1

            target_negative_count = max(1 if split_name == "train" else 0, int(round(len(positive_origins) * negative_ratio)))
            attempts = 0
            made_negatives = 0
            while made_negatives < target_negative_count and attempts < target_negative_count * 50 + 50:
                attempts += 1
                left = rng.randint(0, image_width - tile_size)
                top = rng.randint(0, image_height - tile_size)
                if tile_has_mitosis_center(source_annotations, left, top, tile_size):
                    continue

                tile_file = f"{split_name}/negative/{Path(image_info['file_name']).stem}_neg{made_negatives:02d}_x{left}_y{top}.jpg"
                save_tile(source_image, left, top, tile_size, output_dir / tile_file)
                tile_images.append(
                    {
                        "id": next_image_id,
                        "file_name": tile_file.replace("\\", "/"),
                        "width": tile_size,
                        "height": tile_size,
                        "source_image_id": image_info["id"],
                        "source_file_name": image_info["file_name"],
                        "tile_left": left,
                        "tile_top": top,
                    }
                )
                next_image_id += 1
                negative_tiles += 1
                made_negatives += 1

    tiled_coco = {
        "info": {
            "description": f"512x512 mitotic detection tiles from {coco_path}",
            "tile_size": tile_size,
            "negative_ratio": negative_ratio,
        },
        "images": tile_images,
        "annotations": tile_annotations,
        "categories": [{"id": 1, "name": "mitotic figure"}],
    }
    write_json(output_dir / f"{split_name}_tiles_coco.json", tiled_coco)
    return {
        "split": split_name,
        "tile_images": len(tile_images),
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles,
        "mitotic_annotations": len(tile_annotations),
    }


def build_grid_split(coco_path, output_dir, split_name, tile_size):
    """Tile every source image once on a non-overlapping grid.

    Ground-truth boxes are assigned by center, so each source annotation occurs
    in exactly one evaluation tile. Edge tiles are padded to tile_size.
    """
    coco = load_json(coco_path)
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    tile_images = []
    tile_annotations = []
    next_image_id = 1
    next_annotation_id = 1
    positive_tiles = 0
    negative_tiles = 0
    covered_source_images = set()

    Image.MAX_IMAGE_PIXELS = None
    for image_info in coco["images"]:
        source_path = (coco_path.parent / image_info["file_name"]).resolve()
        source_annotations = annotations_by_image.get(image_info["id"], [])

        with Image.open(source_path) as source_image:
            source_image = source_image.convert("RGB")
            image_width, image_height = source_image.size

            for top in range(0, image_height, tile_size):
                for left in range(0, image_width, tile_size):
                    tile_anns = boxes_for_tile(source_annotations, left, top, tile_size)
                    kind = "positive" if tile_anns else "negative"
                    tile_file = (
                        f"{split_name}/grid/{kind}/"
                        f"{Path(image_info['file_name']).stem}_x{left}_y{top}.jpg"
                    )
                    save_tile(source_image, left, top, tile_size, output_dir / tile_file)
                    tile_images.append(
                        {
                            "id": next_image_id,
                            "file_name": tile_file.replace("\\", "/"),
                            "width": tile_size,
                            "height": tile_size,
                            "source_image_id": image_info["id"],
                            "source_file_name": image_info["file_name"],
                            "source_width": image_width,
                            "source_height": image_height,
                            "tile_left": left,
                            "tile_top": top,
                        }
                    )
                    for tile_ann in tile_anns:
                        tile_ann["id"] = next_annotation_id
                        tile_ann["image_id"] = next_image_id
                        tile_annotations.append(tile_ann)
                        next_annotation_id += 1
                    if tile_anns:
                        positive_tiles += 1
                    else:
                        negative_tiles += 1
                    covered_source_images.add(image_info["id"])
                    next_image_id += 1

    tiled_coco = {
        "info": {
            "description": f"Full-grid mitotic detection tiles from {coco_path}",
            "tile_size": tile_size,
            "strategy": "non_overlapping_full_grid",
        },
        "images": tile_images,
        "annotations": tile_annotations,
        "categories": [{"id": 1, "name": "mitotic figure"}],
    }
    write_json(output_dir / f"{split_name}_tiles_coco.json", tiled_coco)
    return {
        "split": split_name,
        "strategy": "grid",
        "source_images": len(covered_source_images),
        "tile_images": len(tile_images),
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles,
        "mitotic_annotations": len(tile_annotations),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build 512x512 COCO detection tiles from full-image COCO splits.")
    parser.add_argument("--coco-dir", default="demo1/detection_coco/Aperio_CS2", help="Directory with train/test COCO files.")
    parser.add_argument("--output-dir", default="demo1/detection_tiles/Aperio_CS2", help="Output tile dataset directory.")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, help="Tile size.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative tiles per positive tile per image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--eval-strategy",
        choices=["grid", "sampled"],
        default="grid",
        help="Use exhaustive grid tiling for validation/test, or the legacy sampled protocol.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    split_names = [name for name in ["train", "val", "test"] if (coco_dir / f"{name}_coco.json").exists()]
    for split_index, split_name in enumerate(split_names):
        coco_path = coco_dir / f"{split_name}_coco.json"
        if split_name != "train" and args.eval_strategy == "grid":
            summary = build_grid_split(
                coco_path=coco_path,
                output_dir=output_dir,
                split_name=split_name,
                tile_size=args.tile_size,
            )
        else:
            summary = build_split(
                coco_path=coco_path,
                output_dir=output_dir,
                split_name=split_name,
                tile_size=args.tile_size,
                negative_ratio=args.negative_ratio,
                seed=args.seed + split_index * 1000,
            )
        summaries.append(summary)
        print(summary)

    write_json(output_dir / "summary.json", {"splits": summaries})
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
