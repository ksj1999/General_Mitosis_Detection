# Real Pathology Validation Plan

The current work already uses real H&E histopathology images from MIDOG++, but the strongest detector result so far is limited to one held-out scanner:

```text
Train scanners: Hamamatsu S360 + Hamammatsu XR
Held-out scanner: Aperio CS2
```

To show that the method works beyond a small test, use a staged validation story.

## 1. Patch-Level Robustness

Question:

```text
Can stain/color augmentation improve mitotic-vs-non-mitotic candidate classification across scanners?
```

Current leave-one-scanner-out result:

```text
Model             Mean F1   Mean Precision   Mean Recall   Mean Accuracy
pixel_baseline    0.429     0.395            0.508         0.509
pixel_augmented   0.463     0.409            0.564         0.510
```

Interpretation:

```text
Color augmentation improves average held-out scanner patch classification.
```

## 2. Full Detection On Real Image Tiles

Question:

```text
Can stain/color augmentation improve mitotic figure detection in 512 x 512 pathology image tiles?
```

Current held-out Aperio CS2 result:

```text
Model                       Score   Precision   Recall   F1
Faster R-CNN                0.70    0.602       0.389    0.473
Faster R-CNN + color jitter 0.70    0.604       0.727    0.660
```

Interpretation:

```text
Color augmentation substantially improves detector recall and F1 on a held-out scanner while preserving precision.
```

## 3. Next Stronger Validation: Full Detection LOSO

Run full detection for all three held-out scanner settings:

```text
Held-out Aperio CS2
Held-out Hamamatsu S360
Held-out Hamammatsu XR
```

For each holdout:

```text
1. Build detection tiles.
2. Train Faster R-CNN baseline.
3. Train Faster R-CNN + color jitter.
4. Evaluate precision, recall, F1.
```

This lets you claim:

```text
We evaluated scanner generalization using leave-one-scanner-out detection experiments on real H&E pathology image tiles.
```

## Commands

Build tile datasets for every held-out scanner:

```powershell
.\.venv-det\Scripts\python.exe demo1\demo1_build_all_detection_tiles.py
```

Train baseline for a holdout:

```powershell
.\.venv-det\Scripts\python.exe demo1\demo1_train_detection_frcnn.py --device cuda --epochs 5 --batch-size 4 --train-coco demo1\detection_tiles\Aperio_CS2\train_tiles_coco.json --test-coco demo1\detection_tiles\Aperio_CS2\test_tiles_coco.json --output-dir demo1\models\detection_frcnn_Aperio_CS2
```

Train color-jitter model:

```powershell
.\.venv-det\Scripts\python.exe demo1\demo1_train_detection_frcnn.py --device cuda --epochs 5 --batch-size 4 --color-jitter --train-coco demo1\detection_tiles\Aperio_CS2\train_tiles_coco.json --test-coco demo1\detection_tiles\Aperio_CS2\test_tiles_coco.json --output-dir demo1\models\detection_frcnn_Aperio_CS2_color_jitter
```

Evaluate:

```powershell
.\.venv-det\Scripts\python.exe demo1\demo1_evaluate_detection_frcnn.py --device cuda --batch-size 4 --checkpoint demo1\models\detection_frcnn_Aperio_CS2\final.pt --coco demo1\detection_tiles\Aperio_CS2\test_tiles_coco.json --output-dir demo1\models\detection_frcnn_Aperio_CS2\eval
```

## What To Show In A Demo

Use these artifacts:

```text
demo1/outputs/scanner_stain_comparison.jpg
demo1/outputs/detection_tile_positive_preview.jpg
demo1/models/loso_pixel/loso_summary.md
demo1/models/detection_frcnn_aperio_b4/eval/README.md
demo1/models/detection_frcnn_aperio_color_jitter/eval/README.md
```

## Limitations To State Clearly

```text
The current full-detection result is tile-level, not yet whole-slide aggregation.
The strongest detector comparison has been run on Aperio CS2 holdout first.
The next step is full leave-one-scanner-out detector evaluation across all three scanner domains.
```

That is honest and still strong: it shows a real detector on real pathology image tiles, with a clear plan to scale the evidence.
