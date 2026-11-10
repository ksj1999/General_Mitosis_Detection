import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


LABEL_TO_INDEX = {
    2: 0,
    1: 1,
}


def read_slide_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_annotations(path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    images_by_name = {image["file_name"]: image for image in data["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    return images_by_name, annotations_by_image


def centered_crop_box(bbox, patch_size, image_width, image_height):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = int(round(center_x - patch_size / 2))
    top = int(round(center_y - patch_size / 2))
    right = left + patch_size
    bottom = top + patch_size

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    return max(0, left), max(0, top), right, bottom


def parse_args():
    parser = argparse.ArgumentParser(description="Cache all breast-cancer annotation patches for LOSO experiments.")
    parser.add_argument("--subset-dir", default="jun_model/breast_cancer_subset", help="Breast cancer subset directory.")
    parser.add_argument("--patch-size", type=int, default=128, help="Centered square crop size.")
    parser.add_argument("--output", default="jun_model/demo1/patch_dataset_all/breast_patch_cache.npz", help="Output cache.")
    return parser.parse_args()


def main():
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    slide_rows = read_slide_rows(subset_dir / "metadata" / "breast_cancer_slides.csv")
    images_by_name, annotations_by_image = load_annotations(subset_dir / "metadata" / "annotations_breast_cancer.json")

    images = []
    labels = []
    scanners = []
    source_dataset_splits = []
    source_images = []
    annotation_ids = []

    Image.MAX_IMAGE_PIXELS = None

    for slide_index, row in enumerate(slide_rows, start=1):
        file_name = row["file_name"]
        image_info = images_by_name[file_name]
        annotations = annotations_by_image[image_info["id"]]
        image_path = subset_dir / "images" / file_name

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size

            for annotation in annotations:
                left, top, right, bottom = centered_crop_box(
                    annotation["bbox"],
                    patch_size=args.patch_size,
                    image_width=image_width,
                    image_height=image_height,
                )
                patch = image.crop((left, top, right, bottom))
                images.append(np.asarray(patch, dtype=np.uint8))
                labels.append(LABEL_TO_INDEX[int(annotation["category_id"])])
                scanners.append(row["Scanner"])
                source_dataset_splits.append(row["Dataset"])
                source_images.append(file_name)
                annotation_ids.append(annotation["id"])

        if slide_index % 10 == 0:
            print(f"processed {slide_index}/{len(slide_rows)} slides, patches={len(images)}", flush=True)

    np.savez_compressed(
        output,
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        scanners=np.asarray(scanners),
        source_dataset_splits=np.asarray(source_dataset_splits),
        source_images=np.asarray(source_images),
        annotation_ids=np.asarray(annotation_ids, dtype=np.int64),
        patch_size=np.asarray([args.patch_size], dtype=np.int64),
    )

    print(f"cache: {output}", flush=True)
    print(f"patches: {len(images)}", flush=True)


if __name__ == "__main__":
    main()
