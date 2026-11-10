# MIDOG++ Breast Cancer Subset

This folder contains the human breast cancer subset prepared from the MIDOG++ repository for demo and team development work. It keeps one tumor type while preserving scanner variation, which makes it useful for testing cross-scanner robustness.

## Folder Layout

- `images/`: 150 TIFF slide images.
- `metadata/breast_cancer_slides.csv`: one row per image with split, scanner, origin, and species.
- `metadata/annotations_breast_cancer.json`: filtered COCO-style annotations for only these images.
- `metadata/annotation_counts_by_image.csv`: per-image counts for mitotic and not-mitotic annotations.
- `metadata/categories.csv`: label ID mapping.

## Dataset Summary

```text
Images: 150
Annotations: 4435
Tumor type: human breast cancer
Species: Human
Task: mitotic figure detection
```

Scanner and split counts:

```text
Aperio CS2      train 39, test 11
Hamamatsu S360  train 39, test 11
Hamammatsu XR   train 39, test 11
```

Note: `Hamammatsu XR` is spelled this way in the source metadata.

## Labels

```text
category_id = 1: mitotic figure
category_id = 2: not mitotic figure
```

This is an object detection dataset. Each annotation is a bounding box around one candidate figure.

The bounding box format is:

```text
[x1, y1, x2, y2]
```

where:

```text
x1 = left
y1 = top
x2 = right
y2 = bottom
```

Coordinates are pixel coordinates in the corresponding TIFF image.

## How Images Connect To Labels

Use `metadata/annotations_breast_cancer.json`.

The JSON has these main fields:

```text
images
annotations
categories
```

Each image entry looks like:

```json
{
  "file_name": "001.tiff",
  "id": 1,
  "width": 7215,
  "height": 5412,
  "tumor_type": "human breast cancer"
}
```

Each annotation entry looks like:

```json
{
  "bbox": [4336, 346, 4386, 396],
  "category_id": 2,
  "image_id": 1,
  "labels": [2, 2],
  "id": 1
}
```

To get all labels for an image:

```text
annotations[].image_id == images[].id
```

Example:

```text
001.tiff has image id 1.
All annotations with image_id = 1 belong to 001.tiff.
```

## Train/Test Split

The train/test split comes from the original MIDOG++ metadata file, not from this subset preparation.

Use:

```text
metadata/breast_cancer_slides.csv
```

Important columns:

```text
Slide       numeric slide ID
file_name   TIFF filename in images/
Dataset     train or test
Tumor       tumor type
Scanner     scanner domain
Origin      data origin
Species     species
```

The `Slide` number maps to the TIFF filename:

```text
Slide 1 -> 001.tiff
Slide 52 -> 052.tiff
Slide 101 -> 101.tiff
```

## Suggested Demo Split

For a scanner-robustness demo, train on two scanner domains and hold out the third scanner domain.

Example:

```text
Train:
  Hamammatsu XR
  Hamamatsu S360

Test:
  Aperio CS2
```

This tests whether the model generalizes to an unseen scanner while keeping the tumor type fixed.

## Loading Example

```python
import json
from pathlib import Path

root = Path("jun_model/breast_cancer_subset")

with open(root / "metadata" / "annotations_breast_cancer.json", encoding="utf-8") as f:
    data = json.load(f)

images = {image["id"]: image for image in data["images"]}
annotations_by_image = {}

for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    annotations_by_image.setdefault(image_id, []).append(annotation)

image_id = 1
image_info = images[image_id]
image_path = root / "images" / image_info["file_name"]
image_annotations = annotations_by_image.get(image_id, [])

print(image_path)
print(image_info)
print(image_annotations[:3])
```

## Notes

- The TIFF files are large whole-slide or region-of-interest images.
- Some tooling may need OpenSlide or TIFF-compatible image loaders.
- `annotation_counts_by_image.csv` is for quick inspection only; use `annotations_breast_cancer.json` for training labels.
- The source repo also includes SQLite annotations, but this subset uses JSON/CSV because they are easier to share and inspect.

## Packaging Status

```text
Copied images: 150
Missing source images: 0
Filtered annotations: 4435
```
