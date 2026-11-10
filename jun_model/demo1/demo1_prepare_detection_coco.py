import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MITOTIC_CATEGORY_ID = 1


def read_slide_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def coco_bbox(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def build_coco(source, slide_rows, selected_scanners, image_root, selected_datasets=None):
    selected_files = {
        row["file_name"]
        for row in slide_rows
        if row["Scanner"] in selected_scanners
        and (selected_datasets is None or row["Dataset"] in selected_datasets)
    }
    selected_images = [image for image in source["images"] if image["file_name"] in selected_files]
    selected_image_ids = {image["id"] for image in selected_images}

    images = []
    for image in selected_images:
        images.append(
            {
                "id": image["id"],
                "file_name": str(Path(image_root) / image["file_name"]).replace("\\", "/"),
                "width": image["width"],
                "height": image["height"],
                "tumor_type": image.get("tumor_type", "human breast cancer"),
            }
        )

    annotations = []
    annotation_id = 1
    for annotation in source["annotations"]:
        if annotation["image_id"] not in selected_image_ids:
            continue
        if annotation["category_id"] != MITOTIC_CATEGORY_ID:
            continue

        x1, y1, x2, y2 = annotation["bbox"]
        width = float(x2 - x1)
        height = float(y2 - y1)
        if width <= 0 or height <= 0:
            continue

        annotations.append(
            {
                "id": annotation_id,
                "image_id": annotation["image_id"],
                "category_id": 1,
                "bbox": coco_bbox(x1, y1, x2, y2),
                "area": width * height,
                "iscrowd": 0,
                "source_annotation_id": annotation["id"],
            }
        )
        annotation_id += 1

    return {
        "info": {
            "description": "MIDOG++ human breast cancer detection split",
            "task": "mitotic figure detection",
            "bbox_format": "COCO [x, y, width, height]",
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "mitotic figure"}],
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def summarize_split(name, coco):
    image_ids = {image["id"] for image in coco["images"]}
    images_with_annotations = {annotation["image_id"] for annotation in coco["annotations"]}
    return {
        "split": name,
        "images": len(coco["images"]),
        "mitotic_annotations": len(coco["annotations"]),
        "images_with_mitoses": len(images_with_annotations),
        "images_without_mitoses": len(image_ids - images_with_annotations),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare COCO-style detection splits for breast cancer scanner LOSO.")
    parser.add_argument("--subset-dir", default="jun_model/breast_cancer_subset", help="Prepared breast cancer subset directory.")
    parser.add_argument("--output-dir", default="jun_model/demo1/detection_coco", help="Output directory.")
    parser.add_argument(
        "--image-root",
        default="../../../jun_model/breast_cancer_subset/images",
        help="Image root path written into COCO file_name fields.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)

    slide_rows = read_slide_rows(subset_dir / "metadata" / "breast_cancer_slides.csv")
    source = load_json(subset_dir / "metadata" / "annotations_breast_cancer.json")
    scanners = sorted({row["Scanner"] for row in slide_rows})

    summary_rows = []
    for held_out_scanner in scanners:
        train_scanners = [scanner for scanner in scanners if scanner != held_out_scanner]
        split_dir = output_dir / held_out_scanner.replace(" ", "_")

        # Use the source split only within the two training scanners. The held-out
        # scanner remains untouched until the final test evaluation.
        train_coco = build_coco(
            source,
            slide_rows,
            train_scanners,
            args.image_root,
            selected_datasets={"train"},
        )
        val_coco = build_coco(
            source,
            slide_rows,
            train_scanners,
            args.image_root,
            selected_datasets={"test"},
        )
        test_coco = build_coco(source, slide_rows, [held_out_scanner], args.image_root)

        write_json(split_dir / "train_coco.json", train_coco)
        write_json(split_dir / "val_coco.json", val_coco)
        write_json(split_dir / "test_coco.json", test_coco)

        train_summary = summarize_split("train", train_coco)
        val_summary = summarize_split("val", val_coco)
        test_summary = summarize_split("test", test_coco)
        for row in [train_summary, val_summary, test_summary]:
            row["held_out_scanner"] = held_out_scanner
            row["train_scanners"] = ", ".join(train_scanners)
            summary_rows.append(row)

        print(f"held out scanner: {held_out_scanner}")
        print(f"  train scanners: {', '.join(train_scanners)}")
        print(f"  train images: {train_summary['images']}, mitotic boxes: {train_summary['mitotic_annotations']}")
        print(f"  val images: {val_summary['images']}, mitotic boxes: {val_summary['mitotic_annotations']}")
        print(f"  test images: {test_summary['images']}, mitotic boxes: {test_summary['mitotic_annotations']}")

    summary_path = output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "held_out_scanner",
            "train_scanners",
            "split",
            "images",
            "mitotic_annotations",
            "images_with_mitoses",
            "images_without_mitoses",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    total_positive = Counter(annotation["category_id"] for annotation in source["annotations"])[MITOTIC_CATEGORY_ID]
    print(f"summary: {summary_path}")
    print(f"source mitotic annotations: {total_positive}")


if __name__ == "__main__":
    main()
