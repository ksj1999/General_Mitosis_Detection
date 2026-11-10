import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


LABEL_NAMES = {
    1: "mitotic_figure",
    2: "not_mitotic_figure",
}


def load_annotations(annotation_path):
    with annotation_path.open(encoding="utf-8") as f:
        data = json.load(f)

    images_by_name = {image["file_name"]: image for image in data["images"]}
    annotations_by_image = defaultdict(list)

    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    return images_by_name, annotations_by_image


def load_manifest(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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

    left = max(0, left)
    top = max(0, top)
    return left, top, right, bottom


def build_split(split_name, manifest_path, subset_dir, output_dir, images_by_name, annotations_by_image, patch_size):
    rows = load_manifest(manifest_path)
    metadata_rows = []
    patches_written = 0

    Image.MAX_IMAGE_PIXELS = None

    for slide_row in rows:
        file_name = slide_row["file_name"]
        image_info = images_by_name[file_name]
        image_path = subset_dir / "images" / file_name
        annotations = annotations_by_image.get(image_info["id"], [])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size

            for annotation in annotations:
                category_id = int(annotation["category_id"])
                label_name = LABEL_NAMES[category_id]
                left, top, right, bottom = centered_crop_box(
                    annotation["bbox"],
                    patch_size=patch_size,
                    image_width=image_width,
                    image_height=image_height,
                )

                patch = image.crop((left, top, right, bottom))
                patch_name = f"{Path(file_name).stem}_ann{annotation['id']:05d}_cat{category_id}.jpg"
                patch_rel_path = Path(split_name) / label_name / patch_name
                patch_path = output_dir / "patches" / patch_rel_path
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                patch.save(patch_path, quality=95)

                metadata_rows.append(
                    {
                        "split": split_name,
                        "patch_path": str(Path("patches") / patch_rel_path).replace("\\", "/"),
                        "source_image": file_name,
                        "image_id": image_info["id"],
                        "annotation_id": annotation["id"],
                        "category_id": category_id,
                        "label": label_name,
                        "scanner": slide_row["Scanner"],
                        "source_dataset_split": slide_row["Dataset"],
                        "bbox_x1": annotation["bbox"][0],
                        "bbox_y1": annotation["bbox"][1],
                        "bbox_x2": annotation["bbox"][2],
                        "bbox_y2": annotation["bbox"][3],
                        "crop_left": left,
                        "crop_top": top,
                        "crop_right": right,
                        "crop_bottom": bottom,
                    }
                )
                patches_written += 1

    return metadata_rows, patches_written


def write_metadata(path, rows):
    if not rows:
        raise ValueError("No patches were generated.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Build patch-level Demo 1 dataset from bbox annotations.")
    parser.add_argument("--subset-dir", default="jun_model/breast_cancer_subset", help="Path to breast cancer subset.")
    parser.add_argument("--split-dir", default="jun_model/demo1/splits", help="Path to scanner split CSVs.")
    parser.add_argument("--output-dir", default="jun_model/demo1/patch_dataset", help="Output patch dataset directory.")
    parser.add_argument("--patch-size", type=int, default=128, help="Square patch size centered on each bbox.")
    return parser.parse_args()


def main():
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    annotation_path = subset_dir / "metadata" / "annotations_breast_cancer.json"
    images_by_name, annotations_by_image = load_annotations(annotation_path)

    all_rows = []
    for split_name, manifest_name in [
        ("train", "train_scanner_holdout.csv"),
        ("test", "test_scanner_holdout.csv"),
    ]:
        metadata_rows, count = build_split(
            split_name=split_name,
            manifest_path=split_dir / manifest_name,
            subset_dir=subset_dir,
            output_dir=output_dir,
            images_by_name=images_by_name,
            annotations_by_image=annotations_by_image,
            patch_size=args.patch_size,
        )
        all_rows.extend(metadata_rows)
        print(f"{split_name}: wrote {count} patches")

    write_metadata(output_dir / "patch_metadata.csv", all_rows)
    print(f"metadata: {output_dir / 'patch_metadata.csv'}")
    print(f"patch size: {args.patch_size}x{args.patch_size}")


if __name__ == "__main__":
    main()
