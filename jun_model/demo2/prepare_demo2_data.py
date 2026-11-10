import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SCALES = {
    "20x": 0.50,
    "40x": 0.25,
}
OUTPUT_SIZE = 224
LABEL_NAMES = {1: "mitotic", 2: "non_mitotic"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scanner_slug(scanner):
    return re.sub(r"[^a-z0-9]+", "_", scanner.lower()).strip("_")


def resolution_to_mpp(pixels_per_unit, resolution_unit):
    pixels_per_unit = float(pixels_per_unit)
    if pixels_per_unit <= 0:
        raise ValueError(f"Resolution must be positive, got {pixels_per_unit}")
    if resolution_unit == 2:  # inch
        return 25400.0 / pixels_per_unit
    if resolution_unit == 3:  # centimeter
        return 10000.0 / pixels_per_unit
    raise ValueError(f"Unsupported TIFF ResolutionUnit {resolution_unit!r}")


def read_tiff_metadata(path):
    with Image.open(path) as image:
        width, height = image.size
        x_resolution = image.tag_v2.get(282)
        y_resolution = image.tag_v2.get(283)
        resolution_unit = int(image.tag_v2.get(296, 1))
        if x_resolution is None or y_resolution is None:
            raise ValueError(f"Missing TIFF resolution tags in {path}")
        return {
            "width": width,
            "height": height,
            "mpp_x": resolution_to_mpp(x_resolution, resolution_unit),
            "mpp_y": resolution_to_mpp(y_resolution, resolution_unit),
            "resolution_unit": resolution_unit,
        }


def centered_crop_geometry(center_x, center_y, width, height, mpp_x, mpp_y, target_mpp):
    crop_width = max(1, int(round(OUTPUT_SIZE * target_mpp / mpp_x)))
    crop_height = max(1, int(round(OUTPUT_SIZE * target_mpp / mpp_y)))
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    right = left + crop_width
    bottom = top + crop_height
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": crop_width,
        "height": crop_height,
        "touches_edge": left < 0 or top < 0 or right > width or bottom > height,
    }


def point_in_crop(x, y, geometry):
    return geometry["left"] <= x < geometry["right"] and geometry["top"] <= y < geometry["bottom"]


def load_source(subset_dir):
    slide_rows = read_csv(subset_dir / "metadata" / "breast_cancer_slides.csv")
    with (subset_dir / "metadata" / "annotations_breast_cancer.json").open(encoding="utf-8") as handle:
        coco = json.load(handle)
    return slide_rows, coco


def prepare_manifests(subset_dir, output_dir):
    slide_rows, coco = load_source(subset_dir)
    slides_by_file = {row["file_name"]: row for row in slide_rows}
    coco_images = {image["id"]: image for image in coco["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    expected_files = set(slides_by_file)
    json_files = {image["file_name"] for image in coco["images"]}
    if expected_files != json_files:
        raise ValueError(
            f"Slide CSV and annotation JSON disagree: CSV-only={sorted(expected_files - json_files)}, "
            f"JSON-only={sorted(json_files - expected_files)}"
        )

    image_rows = []
    candidate_rows = []
    invalid_geometry = []
    out_of_bounds_boxes = []
    duplicate_keys = Counter()

    for image_id in sorted(coco_images):
        image_info = coco_images[image_id]
        file_name = image_info["file_name"]
        slide = slides_by_file[file_name]
        image_path = subset_dir / "images" / file_name
        metadata = read_tiff_metadata(image_path)
        if (metadata["width"], metadata["height"]) != (image_info["width"], image_info["height"]):
            raise ValueError(f"Dimension mismatch for {file_name}")

        image_row = {
            "image_id": image_id,
            "slide_id": slide["Slide"],
            "file_name": file_name,
            "image_path": image_path.as_posix(),
            "scanner": slide["Scanner"],
            "origin": slide["Origin"],
            "source_dataset_split": slide["Dataset"],
            "width": metadata["width"],
            "height": metadata["height"],
            "mpp_x": f"{metadata['mpp_x']:.6f}",
            "mpp_y": f"{metadata['mpp_y']:.6f}",
            "resolution_unit": metadata["resolution_unit"],
        }
        for scale_name, target_mpp in SCALES.items():
            image_row[f"source_crop_width_{scale_name}_px"] = int(
                round(OUTPUT_SIZE * target_mpp / metadata["mpp_x"])
            )
            image_row[f"source_crop_height_{scale_name}_px"] = int(
                round(OUTPUT_SIZE * target_mpp / metadata["mpp_y"])
            )
        image_rows.append(image_row)

        annotations = annotations_by_image[image_id]
        mitotic_centers = []
        for annotation in annotations:
            x1, y1, x2, y2 = map(float, annotation["bbox"])
            if x2 <= x1 or y2 <= y1:
                invalid_geometry.append(annotation["id"])
            if x1 < 0 or y1 < 0 or x2 > metadata["width"] or y2 > metadata["height"]:
                out_of_bounds_boxes.append(annotation["id"])
            if int(annotation["category_id"]) == 1:
                mitotic_centers.append((annotation["id"], (x1 + x2) / 2, (y1 + y2) / 2))
            duplicate_keys[(image_id, tuple(annotation["bbox"]), annotation["category_id"])] += 1

        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id not in LABEL_NAMES:
                raise ValueError(f"Unexpected category {category_id}")
            x1, y1, x2, y2 = map(float, annotation["bbox"])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            row = {
                "candidate_id": annotation["id"],
                "image_id": image_id,
                "slide_id": slide["Slide"],
                "file_name": file_name,
                "image_path": image_path.as_posix(),
                "scanner": slide["Scanner"],
                "origin": slide["Origin"],
                "source_dataset_split": slide["Dataset"],
                "category_id": category_id,
                "label": 1 if category_id == 1 else 0,
                "label_name": LABEL_NAMES[category_id],
                "bbox_x1": f"{x1:.3f}",
                "bbox_y1": f"{y1:.3f}",
                "bbox_x2": f"{x2:.3f}",
                "bbox_y2": f"{y2:.3f}",
                "center_x": f"{center_x:.3f}",
                "center_y": f"{center_y:.3f}",
                "image_width": metadata["width"],
                "image_height": metadata["height"],
                "mpp_x": f"{metadata['mpp_x']:.6f}",
                "mpp_y": f"{metadata['mpp_y']:.6f}",
                "bbox_outside_image": x1 < 0
                or y1 < 0
                or x2 > metadata["width"]
                or y2 > metadata["height"],
                "duplicate_exact_annotation": duplicate_keys[
                    (image_id, tuple(annotation["bbox"]), annotation["category_id"])
                ]
                > 1,
            }
            for scale_name, target_mpp in SCALES.items():
                geometry = centered_crop_geometry(
                    center_x,
                    center_y,
                    metadata["width"],
                    metadata["height"],
                    metadata["mpp_x"],
                    metadata["mpp_y"],
                    target_mpp,
                )
                other_mitoses = sum(
                    mitosis_id != annotation["id"] and point_in_crop(mx, my, geometry)
                    for mitosis_id, mx, my in mitotic_centers
                )
                row[f"crop_left_{scale_name}"] = geometry["left"]
                row[f"crop_top_{scale_name}"] = geometry["top"]
                row[f"crop_right_{scale_name}"] = geometry["right"]
                row[f"crop_bottom_{scale_name}"] = geometry["bottom"]
                row[f"touches_edge_{scale_name}"] = geometry["touches_edge"]
                row[f"other_mitoses_{scale_name}"] = other_mitoses
                row[f"ambiguous_negative_{scale_name}"] = category_id == 2 and other_mitoses > 0
            candidate_rows.append(row)

    if invalid_geometry:
        raise ValueError(f"Invalid annotation geometry: {invalid_geometry[:10]}")

    duplicate_groups = sum(count > 1 for count in duplicate_keys.values())
    write_csv(output_dir / "images.csv", image_rows)
    write_csv(output_dir / "candidates.csv", candidate_rows)

    scanners = sorted({row["scanner"] for row in image_rows})
    fold_image_rows = []
    fold_candidate_rows = []
    summary_counter = Counter()
    for held_out_scanner in scanners:
        fold = scanner_slug(held_out_scanner)
        image_roles = {}
        for image_row in image_rows:
            if image_row["scanner"] == held_out_scanner:
                role = "test"
            elif image_row["source_dataset_split"] == "train":
                role = "train"
            elif image_row["source_dataset_split"] == "test":
                role = "val"
            else:
                raise ValueError(f"Unknown source split {image_row['source_dataset_split']!r}")
            image_roles[int(image_row["image_id"])] = role
            fold_image_rows.append(
                {
                    "fold": fold,
                    "held_out_scanner": held_out_scanner,
                    "image_id": image_row["image_id"],
                    "file_name": image_row["file_name"],
                    "scanner": image_row["scanner"],
                    "role": role,
                }
            )
            summary_counter[(fold, held_out_scanner, role, image_row["scanner"], "images", "all")] += 1

        for candidate_row in candidate_rows:
            role = image_roles[int(candidate_row["image_id"])]
            fold_candidate_rows.append(
                {
                    "fold": fold,
                    "held_out_scanner": held_out_scanner,
                    "candidate_id": candidate_row["candidate_id"],
                    "image_id": candidate_row["image_id"],
                    "scanner": candidate_row["scanner"],
                    "role": role,
                    "label": candidate_row["label"],
                    "label_name": candidate_row["label_name"],
                }
            )
            summary_counter[
                (fold, held_out_scanner, role, candidate_row["scanner"], "candidates", candidate_row["label_name"])
            ] += 1

    write_csv(output_dir / "fold_images.csv", fold_image_rows)
    write_csv(output_dir / "fold_candidates.csv", fold_candidate_rows)
    summary_rows = [
        {
            "fold": key[0],
            "held_out_scanner": key[1],
            "role": key[2],
            "scanner": key[3],
            "unit": key[4],
            "label_name": key[5],
            "count": count,
        }
        for key, count in sorted(summary_counter.items())
    ]
    write_csv(output_dir / "split_summary.csv", summary_rows)

    audit = {
        "images": len(image_rows),
        "candidates": len(candidate_rows),
        "labels": dict(Counter(row["label_name"] for row in candidate_rows)),
        "scanners": dict(Counter(row["scanner"] for row in image_rows)),
        "duplicate_exact_annotation_groups": duplicate_groups,
        "bbox_outside_image": len(out_of_bounds_boxes),
        "ambiguous_negative_20x": sum(row["ambiguous_negative_20x"] for row in candidate_rows),
        "ambiguous_negative_40x": sum(row["ambiguous_negative_40x"] for row in candidate_rows),
        "edge_candidates_20x": sum(row["touches_edge_20x"] for row in candidate_rows),
        "edge_candidates_40x": sum(row["touches_edge_40x"] for row in candidate_rows),
        "mpp_by_scanner": {},
    }
    for scanner in scanners:
        scanner_images = [row for row in image_rows if row["scanner"] == scanner]
        mpp_values = [float(row["mpp_x"]) for row in scanner_images]
        audit["mpp_by_scanner"][scanner] = {
            "min": min(mpp_values),
            "max": max(mpp_values),
            "mean": sum(mpp_values) / len(mpp_values),
        }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare verified Demo 2 candidate and LOSO manifests.")
    parser.add_argument("--subset-dir", type=Path, default=Path("jun_model/breast_cancer_subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("jun_model/demo2/data"))
    return parser.parse_args()


def main():
    args = parse_args()
    audit = prepare_manifests(args.subset_dir, args.output_dir)
    print(json.dumps(audit, indent=2), flush=True)
    print(f"output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
