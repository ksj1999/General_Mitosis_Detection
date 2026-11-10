# MIDOG++ Scanner Generalization Progress Report

## Project Goal

The goal of this project is to test whether mitotic figure detection can generalize across pathology scanner domains. The main practical question is:

```text
If a model is trained on breast cancer images from some scanners, how well does it detect mitotic figures on images from a different scanner?
```

This matters because pathology images can look different across scanners due to color, stain appearance, contrast, brightness, and acquisition differences. A model that works only on one scanner may not be reliable in real pathology workflows.

## Dataset Used

The work uses the MIDOG++ human breast cancer subset prepared in:

```text
jun_model/breast_cancer_subset/
```

The subset contains one tumor type while preserving scanner variation.

### Subset Contents

```text
Images: 150 TIFF images
Annotations: 4435 object annotations
Tumor type: human breast cancer
Species: Human
Task: mitotic figure detection
```

### Scanner Split

Each scanner has 50 images:

| Scanner | Train images | Test images | Total |
| --- | ---: | ---: | ---: |
| Aperio CS2 | 39 | 11 | 50 |
| Hamamatsu S360 | 39 | 11 | 50 |
| Hamammatsu XR | 39 | 11 | 50 |

Note: `Hamammatsu XR` is spelled this way in the source metadata.

### Important Metadata Files

| File | Purpose |
| --- | --- |
| `jun_model/breast_cancer_subset/metadata/breast_cancer_slides.csv` | Image-level metadata: filename, split, scanner, tumor type, origin, species |
| `jun_model/breast_cancer_subset/metadata/annotations_breast_cancer.json` | COCO-style object detection labels |
| `jun_model/breast_cancer_subset/metadata/categories.csv` | Label ID mapping |
| `jun_model/breast_cancer_subset/metadata/annotation_counts_by_image.csv` | Quick count of annotations per image |

The detection labels are bounding boxes in this format:

```text
[x1, y1, x2, y2]
```

The labels are:

```text
category_id = 1: mitotic figure
category_id = 2: not mitotic figure
```

For model training, the current detection experiments use mitotic figure boxes as the positive detection target.

## Main Experimental Setup

The project uses a leave-one-scanner-out setup.

For each run:

```text
Train on two scanner domains.
Test on the held-out scanner domain.
```

This directly tests scanner generalization.

Example:

```text
Train:
  Aperio CS2
  Hamammatsu XR

Test:
  Hamamatsu S360
```

The shorthand `LOSO` means:

```text
Leave-One-Scanner-Out
```

## Work Completed

## 1. Prepared A Shareable Breast Cancer Subset

A new folder was created:

```text
jun_model/breast_cancer_subset/
```

It includes the breast cancer TIFF images and filtered metadata/annotations so teammates can work without needing to inspect the full original MIDOG++ structure.

The README explains:

- how images connect to labels
- where the metadata is stored
- what each label means
- how the train/test split is represented
- how to load the JSON annotations in Python

Main README:

```text
jun_model/breast_cancer_subset/README.md
```

## 2. Built Patch-Level Experiments

Before training a full detector, patch-level experiments were run to quickly test whether stain/color augmentation helps scanner generalization.

These experiments used cropped patches around candidate regions and compared:

```text
pixel_baseline
pixel_augmented
```

The patch-level summary is saved at:

```text
jun_model/demo1/models/loso_pixel/loso_summary.md
```

### Patch-Level Results

| Held-out scanner | Pixel F1 | Augmented F1 | Delta F1 | Pixel recall | Augmented recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aperio CS2 | 0.412 | 0.450 | +0.038 | 0.384 | 0.440 |
| Hamamatsu S360 | 0.346 | 0.413 | +0.068 | 0.345 | 0.466 |
| Hamammatsu XR | 0.529 | 0.525 | -0.004 | 0.794 | 0.785 |

### Mean Patch-Level Results

| Model | Mean F1 | Mean precision | Mean recall | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| pixel_baseline | 0.429 | 0.395 | 0.508 | 0.509 |
| pixel_augmented | 0.463 | 0.409 | 0.564 | 0.510 |

### Patch-Level Interpretation

Color/stain augmentation improved mean held-out F1 from `0.429` to `0.463` and mean recall from `0.508` to `0.564`.

This suggested that scanner/stain robustness is a reasonable direction. However, this was still patch classification, not full object detection.

## 3. Built Full Detection Tile Datasets

The next step was to move from patch classification to object detection.

Large TIFF images were converted into smaller detection tiles so Faster R-CNN could train on them. Positive tiles contain mitotic annotations, and negative tiles provide background/non-mitotic regions.

Tile generation script:

```text
jun_model/demo1/demo1_build_detection_tiles.py
```

All-scanner tile generation script:

```text
jun_model/demo1/demo1_build_all_detection_tiles.py
```

### Detection Tile Counts

| Scanner | Train tile images | Test tile images | Train positive tiles | Test positive tiles |
| --- | ---: | ---: | ---: | ---: |
| Aperio CS2 | 2078 | 1376 | 1033 | 688 |
| Hamamatsu S360 | 2287 | 1164 | 1139 | 582 |
| Hamammatsu XR | 2549 | 902 | 1270 | 451 |

The generated tile datasets are stored in:

```text
jun_model/demo1/detection_tiles/
```

## 4. Trained Full Object Detection Models

The detection model used was:

```text
Faster R-CNN with ResNet50 FPN
```

Main training script:

```text
jun_model/demo1/demo1_train_detection_frcnn.py
```

Main evaluation script:

```text
jun_model/demo1/demo1_evaluate_detection_frcnn.py
```

Checkpoint sweep script:

```text
jun_model/demo1/demo1_evaluate_detection_checkpoints.py
```

All LOSO detection script:

```text
jun_model/demo1/demo1_run_all_detection_loso.py
```

### Training Stabilization

Early detection runs showed that the model could train, but results varied across epochs. Training was then stabilized with:

```text
learning rate = 1e-4
gradient clipping = 1.0
checkpoint evaluation across epochs
```

This was important because the best model was not always the final epoch.

## 5. Full Detection LOSO Results

The main stable detection summary is saved at:

```text
jun_model/demo1/models/detection_loso_summary_lr1e4_clip1/checkpoint_sweep_summary.md
```

### Stable Baseline vs Strong Color Jitter

| Held-out scanner | Model | Best checkpoint | Score threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aperio CS2 | baseline | epoch 4 | 0.500 | 0.775 | 0.651 | 0.708 | 604 | 175 | 324 |
| Aperio CS2 | strong color jitter | epoch 7 | 0.300 | 0.731 | 0.666 | 0.697 | 618 | 227 | 310 |
| Hamamatsu S360 | baseline | epoch 6 | 0.700 | 0.633 | 0.817 | 0.714 | 655 | 379 | 147 |
| Hamamatsu S360 | strong color jitter | epoch 2 | 0.700 | 0.718 | 0.642 | 0.678 | 515 | 202 | 287 |
| Hamammatsu XR | baseline | epoch 8 | 0.700 | 0.729 | 0.798 | 0.762 | 419 | 156 | 106 |
| Hamammatsu XR | strong color jitter | epoch 6 | 0.700 | 0.658 | 0.790 | 0.718 | 415 | 216 | 110 |

### Mean Best-Checkpoint Detection Results

| Model | Mean precision | Mean recall | Mean F1 |
| --- | ---: | ---: | ---: |
| baseline | 0.713 | 0.755 | 0.728 |
| strong color jitter | 0.702 | 0.700 | 0.698 |

### Detection Interpretation

The full object detector gave stronger results than the patch classifier, but strong color jitter did not improve detection. The stable baseline was better overall:

```text
Mean F1 baseline: 0.728
Mean F1 strong color jitter: 0.698
```

This suggests that naive strong color augmentation may hurt localization or confidence calibration, even if augmentation helped the simpler patch-level classifier.

## 6. Tested Mild Color Jitter On One Scanner

Because strong color jitter hurt performance, a milder augmentation setting was tested on the Hamamatsu S360 held-out experiment.

Mild jitter used smaller color changes:

```text
brightness = 0.08
contrast = 0.08
saturation = 0.05
hue = 0.01
```

The mild jitter run is saved at:

```text
jun_model/demo1/models/detection_frcnn_Hamamatsu_S360_color_jitter_mild_lr1e4_clip1/
```

### Hamamatsu S360 Comparison

| Method | Best checkpoint | Score threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | epoch 6 | 0.700 | 0.633 | 0.817 | 0.714 | 655 | 379 | 147 |
| strong color jitter | epoch 2 | 0.700 | 0.718 | 0.642 | 0.678 | 515 | 202 | 287 |
| mild color jitter | epoch 6 | 0.700 | 0.641 | 0.734 | 0.684 | 589 | 330 | 213 |

### Mild Jitter Interpretation

Mild color jitter was slightly better than strong color jitter, but still worse than the stable baseline.

Compared with the baseline, mild jitter had:

- slightly higher precision
- lower recall
- lower F1

This means it made fewer false positive detections but missed more true mitotic figures.

## 7. Generated Visual Error Analysis Panels

To understand model behavior, visual comparison panels were generated for the Hamamatsu S360 baseline vs mild jitter models.

Script:

```text
jun_model/demo1/demo1_make_detection_comparison_panels.py
```

Output folder:

```text
jun_model/demo1/outputs/detection_error_panels_s360_baseline_vs_mild/
```

Panel types:

```text
baseline_fp_extra_*.jpg
comparison_fp_extra_*.jpg
baseline_fn_extra_*.jpg
comparison_fn_extra_*.jpg
```

Color coding:

```text
green = ground truth
red = baseline prediction
blue = comparison model prediction
```

These panels are useful for the presentation because they show concrete examples of false positives and false negatives, not only metrics.

## 8. Analyzed Threshold Tradeoffs

A threshold analysis was created for Hamamatsu S360 baseline vs mild jitter.

Script:

```text
jun_model/demo1/demo1_summarize_threshold_tradeoffs.py
```

Output:

```text
jun_model/demo1/outputs/threshold_tradeoffs_s360_baseline_vs_mild/threshold_tradeoffs.md
```

### Threshold Results

| Model | Score | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.05 | 0.195 | 0.974 | 0.325 | 781 | 3230 | 21 |
| baseline | 0.10 | 0.252 | 0.956 | 0.399 | 767 | 2274 | 35 |
| baseline | 0.20 | 0.348 | 0.926 | 0.506 | 743 | 1392 | 59 |
| baseline | 0.30 | 0.422 | 0.903 | 0.575 | 724 | 992 | 78 |
| baseline | 0.50 | 0.535 | 0.854 | 0.658 | 685 | 595 | 117 |
| baseline | 0.70 | 0.633 | 0.817 | 0.714 | 655 | 379 | 147 |
| mild jitter | 0.05 | 0.235 | 0.930 | 0.375 | 746 | 2432 | 56 |
| mild jitter | 0.10 | 0.303 | 0.913 | 0.455 | 732 | 1687 | 70 |
| mild jitter | 0.20 | 0.389 | 0.884 | 0.541 | 709 | 1112 | 93 |
| mild jitter | 0.30 | 0.450 | 0.855 | 0.590 | 686 | 839 | 116 |
| mild jitter | 0.50 | 0.541 | 0.809 | 0.648 | 649 | 551 | 153 |
| mild jitter | 0.70 | 0.641 | 0.734 | 0.684 | 589 | 330 | 213 |

### Threshold Interpretation

Lower thresholds increase recall but create many false positives. Higher thresholds reduce false positives but miss more mitotic figures.

The best F1 for both baseline and mild jitter occurred at:

```text
score threshold = 0.70
```

## 9. Tested Reinhard-Style RGB Normalization

After color jitter did not clearly improve detection, stain/color normalization was tested.

A simple Reinhard-style RGB normalization option was added:

```text
--normalization reinhard_rgb
```

This normalization adjusts each tile toward a target RGB mean and standard deviation.

The option was added to:

```text
jun_model/demo1/detection_dataset.py
jun_model/demo1/demo1_train_detection_frcnn.py
jun_model/demo1/demo1_evaluate_detection_frcnn.py
jun_model/demo1/demo1_run_all_detection_loso.py
jun_model/demo1/demo1_evaluate_detection_checkpoints.py
```

### Important Correction

The first normalization evaluation gave an invalid near-zero result because the model was trained with normalization but evaluated without applying the same normalization.

This was fixed by updating:

```text
jun_model/demo1/demo1_evaluate_detection_checkpoints.py
```

so that checkpoint evaluation correctly passes:

```text
--normalization reinhard_rgb
```

After the fix, the normalization result became competitive.

### Reinhard Normalization Result On Hamamatsu S360

Summary:

```text
jun_model/demo1/models/detection_loso_summary_reinhard_lr1e4_clip1/checkpoint_sweep_summary.md
```

| Method | Best checkpoint | Score threshold | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable baseline | epoch 6 | 0.700 | 0.633 | 0.817 | 0.714 | 655 | 379 | 147 |
| Reinhard RGB normalization | epoch 7 | 0.500 | 0.709 | 0.721 | 0.715 | 578 | 237 | 224 |

### Normalization Interpretation

Reinhard RGB normalization was essentially tied with the stable baseline:

```text
Baseline F1: 0.714
Reinhard RGB normalization F1: 0.715
```

It did not clearly beat the baseline, but it changed the error balance:

```text
Normalization improved precision and reduced false positives.
Normalization reduced recall and missed more true mitotic figures.
```

This is promising because it suggests stain normalization can preserve performance while making detections more conservative.

## Validation Protocol Audit

The results below were produced with the original exploratory protocol. A later audit identified that checkpoint epochs and score thresholds were selected using the held-out scanner, and that test tiles were centered on ground-truth mitoses. Those scores are useful for development comparisons but should not be presented as unbiased full-image generalization results.

The corrected protocol now:

- splits the two training scanners into train and validation slides
- uses validation data to select the checkpoint and confidence threshold
- covers every held-out slide with a deterministic non-overlapping grid
- includes held-out slides with no mitoses
- assigns each source annotation to one evaluation tile
- uses one-to-one matching against the best unmatched ground truth
- checks train/evaluation normalization compatibility
- seeds initialization, shuffling, and augmentation

New experiments must be run before replacing the historical values below. Use:

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_prepare_detection_coco.py
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_build_all_detection_tiles.py
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_run_all_detection_loso.py --device cuda --epochs 8 --batch-size 1 --lr 1e-4 --clip-grad-norm 1.0 --seed 42 --skip-build-tiles --output-suffix _locked_v2 --summary-dir jun_model\demo1\models\detection_loso_summary_locked_v2
```

## Current Main Results

## Patch Classification

| Experiment | Result |
| --- | --- |
| Baseline mean F1 | 0.429 |
| Augmented mean F1 | 0.463 |
| Change | +0.034 F1 |

Patch-level augmentation helped.

## Full Detection

| Experiment | Mean or held-out F1 | Main takeaway |
| --- | ---: | --- |
| Stable Faster R-CNN baseline, all scanners | 0.728 mean F1 | Strongest overall full-detection result |
| Strong color jitter, all scanners | 0.698 mean F1 | Hurt detection performance |
| Mild color jitter, S360 only | 0.684 F1 | Better than strong jitter, worse than baseline |
| Reinhard RGB normalization, S360 only | 0.715 F1 | Tied baseline, fewer false positives, lower recall |

## Overall Interpretation

The project shows that scanner generalization is a real issue and that robustness methods affect precision and recall differently.

The most important finding so far is:

```text
Color augmentation helped patch classification but did not improve full object detection.
```

This is a useful result because it shows that patch-level improvements do not automatically transfer to real detection performance.

The second important finding is:

```text
Simple stain normalization produced competitive full-detection performance on Hamamatsu S360.
```

Normalization did not clearly improve F1, but it reduced false positives and preserved overall performance. This makes it a stronger candidate than color jitter for the next stage.

## Recommended Next Steps

## 1. Run Normalization On All Held-Out Scanners

So far, Reinhard RGB normalization was tested only on Hamamatsu S360.

Next, run it for:

```text
Aperio CS2
Hamammatsu XR
```

This will show whether normalization helps generally or only on one scanner.

## 2. Compare Error Panels For Baseline vs Normalization

Generate visual panels like the baseline vs mild jitter panels, but compare:

```text
baseline vs Reinhard RGB normalization
```

This will help explain what normalization changes visually.

## 3. Add A Better Stain Normalization Method

The current normalization is a simple RGB mean/std method. A more pathology-specific method may work better, such as:

```text
Macenko normalization
Vahadane normalization
histology-specific stain augmentation
```

This could become a stronger project contribution.

## 4. Consider Domain-Adversarial Learning Later

Adversarial learning is still a possible direction, but it is more complex. Based on the current results, stain normalization should be completed first because it is easier to explain, easier to implement, and already competitive.

## Useful Commands

### Run Stable Detection LOSO

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_run_all_detection_loso.py --device cuda --epochs 8 --batch-size 1 --lr 1e-4 --clip-grad-norm 1.0 --output-suffix _lr1e4_clip1
```

### Evaluate Detection Checkpoints

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_evaluate_detection_checkpoints.py --device cuda --batch-size 4 --output-suffix _lr1e4_clip1 --force
```

### Run Reinhard RGB Normalization For One Scanner

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_train_detection_frcnn.py --device cuda --epochs 8 --batch-size 4 --lr 1e-4 --clip-grad-norm 1.0 --normalization reinhard_rgb --train-coco jun_model\demo1\detection_tiles\Hamamatsu_S360\train_tiles_coco.json --test-coco jun_model\demo1\detection_tiles\Hamamatsu_S360\val_tiles_coco.json --output-dir jun_model\demo1\models\detection_frcnn_Hamamatsu_S360_reinhard_lr1e4_clip1
```

### Select On Validation And Evaluate Reinhard RGB Normalization

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_select_and_test_detection.py --device cuda --batch-size 4 --model-dir jun_model\demo1\models\detection_frcnn_Hamamatsu_S360_reinhard_lr1e4_clip1 --val-coco jun_model\demo1\detection_tiles\Hamamatsu_S360\val_tiles_coco.json --test-coco jun_model\demo1\detection_tiles\Hamamatsu_S360\test_tiles_coco.json --normalization reinhard_rgb --output-dir jun_model\demo1\models\detection_frcnn_Hamamatsu_S360_reinhard_lr1e4_clip1\locked_protocol
```

### Generate Visual Error Panels

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_make_detection_comparison_panels.py
```

## Files To Show Teammates

| File or folder | Why it matters |
| --- | --- |
| `jun_model/breast_cancer_subset/README.md` | Explains the prepared data |
| `jun_model/breast_cancer_subset/metadata/breast_cancer_slides.csv` | Scanner and split metadata |
| `jun_model/breast_cancer_subset/metadata/annotations_breast_cancer.json` | Detection labels |
| `jun_model/demo1/models/loso_pixel/loso_summary.md` | Patch-level results |
| `jun_model/demo1/models/detection_loso_summary_lr1e4_clip1/checkpoint_sweep_summary.md` | Main detection results |
| `jun_model/demo1/models/detection_loso_summary_reinhard_lr1e4_clip1/checkpoint_sweep_summary.md` | Normalization result |
| `jun_model/demo1/outputs/detection_error_panels_s360_baseline_vs_mild/` | Visual false positive/false negative examples |
| `jun_model/demo1/outputs/threshold_tradeoffs_s360_baseline_vs_mild/threshold_tradeoffs.md` | Precision/recall threshold analysis |

## Short Presentation Summary

We prepared a breast cancer subset from MIDOG++ with three scanner domains and used leave-one-scanner-out evaluation to test scanner generalization. Patch-level color augmentation improved F1, but full Faster R-CNN detection showed that strong and mild color jitter did not improve detection performance. A stable baseline reached `0.728` mean F1 across scanners. On Hamamatsu S360, Reinhard-style RGB normalization reached `0.715` F1, essentially matching the baseline while reducing false positives. The next step is to test normalization across all scanners and compare visual error patterns.
