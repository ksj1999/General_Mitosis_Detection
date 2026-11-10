# Demo 1: Breast Cancer Scanner Holdout

This folder contains the first project checkpoint: verify the breast cancer subset and define a scanner-held-out split.

For the current "real pathology" validation story, see:

```text
jun_model/demo1/REAL_PATHOLOGY_VALIDATION.md
```

## Step 1: Visualize Labels

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_visualize_annotations.py --image 001.tiff
```

Output:

```text
jun_model/demo1/outputs/001_annotations.jpg
```

Red boxes are `category_id = 1`, mitotic figure. Blue boxes are `category_id = 2`, not mitotic figure.

## Step 2: Create Scanner Holdout Split

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_make_scanner_split.py
```

Outputs:

```text
jun_model/demo1/splits/train_scanner_holdout.csv
jun_model/demo1/splits/test_scanner_holdout.csv
```

Default split:

```text
Train:
  Hamammatsu XR
  Hamamatsu S360

Test:
  Aperio CS2
```

The split keeps the tumor type fixed as human breast cancer and tests generalization to an unseen scanner.

## Step 3: Build Patch Dataset

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_build_patch_dataset.py
```

Output:

```text
jun_model/demo1/patch_dataset/
  patch_metadata.csv
  patches/
    train/
      mitotic_figure/
      not_mitotic_figure/
    test/
      mitotic_figure/
      not_mitotic_figure/
```

The patch dataset crops a 128 x 128 pixel patch centered on each annotation bbox. This is a controlled patch-level task, not full-image detection.

Generated patch counts:

```text
train mitotic_figure:      834
train not_mitotic_figure: 1421
test mitotic_figure:       688
test not_mitotic_figure:   924
total patches:            3867
```

Scanner breakdown:

```text
train Hamammatsu XR:   943
train Hamamatsu S360: 1312
test Aperio CS2:      1612
```

Use `patch_metadata.csv` as the source of truth for patch paths, labels, original images, scanner domains, and source bbox coordinates.

## Next Step

Train a plain patch classifier as the first baseline. This should answer a limited but useful question: can a model trained on candidate patches from two scanners classify mitotic vs not-mitotic candidates from a held-out scanner?

## Step 4: Fast Feature Baseline

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_train_feature_baseline.py
```

This baseline uses simple RGB/intensity summary features from each patch and trains a weighted logistic classifier. It is not the final model, but it is fast and useful as a first scanner-shift check.

Output:

```text
jun_model/demo1/models/feature_baseline/metrics.json
jun_model/demo1/models/feature_baseline/model.npz
```

Current result:

```text
Train:
  accuracy  0.630
  precision 0.500
  recall    0.647
  F1        0.564

Held-out scanner test, Aperio CS2:
  accuracy  0.587
  precision 0.589
  recall    0.106
  F1        0.180
```

The low held-out scanner recall means this simple color/statistics baseline misses most mitotic figures on the unseen scanner. That is useful evidence that scanner/stain shift matters, but it is only a patch-level result.

## Step 5: Cache Patches

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_cache_patch_dataset.py
```

Output:

```text
jun_model/demo1/patch_dataset/patch_cache.npz
```

This stores all patch images and labels in one NumPy file so later experiments do not repeatedly reopen thousands of JPG files.

## Step 6: Downsampled Pixel Baseline

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_train_pixel_baseline.py
```

This baseline downsamples each 128 x 128 patch to 32 x 32 RGB pixels, flattens the pixels, and trains a weighted logistic classifier.

Output:

```text
jun_model/demo1/models/pixel_baseline/metrics.json
jun_model/demo1/models/pixel_baseline/model.npz
```

Current result:

```text
Train:
  accuracy  0.791
  precision 0.691
  recall    0.788
  F1        0.736

Held-out scanner test, Aperio CS2:
  accuracy  0.535
  precision 0.443
  recall    0.343
  F1        0.387
```

Compared with the simple feature baseline, the pixel baseline learns more morphology and improves held-out scanner F1, but there is still a clear train-to-held-out gap.

## Step 7: Pixel Baseline With Stain/Color Augmentation

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_train_pixel_baseline.py --augment-copies 3 --output-dir jun_model\demo1\models\pixel_augmented_baseline
```

This adds three randomly color-shifted copies of each training patch using brightness, contrast, RGB channel scaling, and RGB channel shift. The held-out scanner test set is not augmented.

Output:

```text
jun_model/demo1/models/pixel_augmented_baseline/metrics.json
jun_model/demo1/models/pixel_augmented_baseline/model.npz
```

Current result:

```text
Train:
  accuracy  0.672
  precision 0.547
  recall    0.669
  F1        0.602

Held-out scanner test, Aperio CS2:
  accuracy  0.533
  precision 0.447
  recall    0.401
  F1        0.423
```

Comparison:

```text
Model                   Train F1   Test F1   Test Precision   Test Recall
Feature baseline        0.564      0.180     0.589            0.106
Pixel baseline          0.736      0.387     0.443            0.343
Pixel + stain/color aug 0.602      0.423     0.447            0.401
```

The augmented pixel baseline improves held-out scanner F1 and recall compared with the non-augmented pixel baseline. This supports the project idea that stain/color augmentation can improve cross-scanner robustness, at least in this patch-level experiment.

To print the comparison table:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\compare_demo1_results.py
```

## Step 8: Leave-One-Scanner-Out Experiment

Build a cache from all 150 breast-cancer slides:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_cache_all_breast_patches.py
```

Run the leave-one-scanner-out pixel experiments:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_run_loso_pixel_experiments.py
```

Outputs:

```text
jun_model/demo1/patch_dataset_all/breast_patch_cache.npz
jun_model/demo1/models/loso_pixel/loso_results.csv
jun_model/demo1/models/loso_pixel/loso_results.json
```

Print the summary:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\compare_loso_results.py
```

Create a report-ready Markdown summary:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\summarize_loso_results.py
```

Current LOSO result:

```text
Held-out scanner | Pixel F1 | Aug F1 | Delta F1 | Pixel Recall | Aug Recall
Aperio CS2       | 0.412    | 0.450  | +0.038   | 0.384        | 0.440
Hamamatsu S360   | 0.346    | 0.413  | +0.068   | 0.345        | 0.466
Hamammatsu XR    | 0.529    | 0.525  | -0.004   | 0.794        | 0.785
```

Stain/color augmentation improves held-out F1 for two of the three scanner holdouts and is roughly flat for `Hamammatsu XR`. This is a stronger Demo 1 result than testing only one held-out scanner.

Mean held-out performance:

```text
Model             Mean F1   Mean Precision   Mean Recall   Mean Accuracy
pixel_baseline    0.429     0.395            0.508         0.509
pixel_augmented   0.463     0.409            0.564         0.510
```

Overall, stain/color augmentation changes mean held-out F1 from `0.429` to `0.463` and mean held-out recall from `0.508` to `0.564`.

## Step 9: Scanner-Statistics Augmentation

Run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_run_loso_scanner_stat_aug.py
```

Print the comparison:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\compare_scanner_stat_aug_results.py
```

Current result:

```text
Held-out scanner | Baseline F1 | Generic Aug F1 | Scanner-Stat F1
Aperio CS2       | 0.412       | 0.450          | 0.415
Hamamatsu S360   | 0.346       | 0.413          | 0.384
Hamammatsu XR    | 0.529       | 0.525          | 0.529
Mean             | 0.429       | 0.463          | 0.442
```

Scanner-statistics augmentation used scanner RGB means to shift training patches toward scanner-like color profiles. It improved over baseline on average, but generic color augmentation still performed best.

## Step 10: Move Toward Full Detection

Prepare COCO-style full-image detection splits:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_prepare_detection_coco.py
```

Output:

```text
jun_model/demo1/detection_coco/
  Aperio_CS2/train_coco.json
  Aperio_CS2/test_coco.json
  Hamamatsu_S360/train_coco.json
  Hamamatsu_S360/test_coco.json
  Hamammatsu_XR/train_coco.json
  Hamammatsu_XR/test_coco.json
  summary.csv
```

These files use only mitotic figure boxes as positive detection targets:

```text
category_id = 1: mitotic figure
```

The original non-mitotic candidate boxes are not included as detector targets.

Visualize one detection target image:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_visualize_detection_coco.py --coco jun_model\demo1\detection_coco\Aperio_CS2\test_coco.json --image-index 0 --output jun_model\demo1\outputs\detection_aperio_test_preview.jpg
```

## Step 11: Build Detection Tiles

Build 512 x 512 detector tiles for the first holdout split:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_build_detection_tiles.py --coco-dir jun_model\demo1\detection_coco\Aperio_CS2 --output-dir jun_model\demo1\detection_tiles\Aperio_CS2
```

Output:

```text
jun_model/demo1/detection_tiles/Aperio_CS2/
  train/positive/
  train/negative/
  test/positive/
  test/negative/
  train_tiles_coco.json
  test_tiles_coco.json
```

Current Aperio holdout tile counts:

```text
Train:
  tile images: 2078
  positive tiles: 1033
  negative tiles: 1045
  mitotic annotations: 1327

Test:
  tile images: 1376
  positive tiles: 688
  negative tiles: 688
  mitotic annotations: 928
```

Visualize a positive tile:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_visualize_detection_tile.py --coco jun_model\demo1\detection_tiles\Aperio_CS2\train_tiles_coco.json --image-index 1 --output jun_model\demo1\outputs\detection_tile_positive_preview.jpg
```

Smoke-test the detection dataset loader:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_detection_loader_smoke_test.py
```

Smoke-test a Faster R-CNN forward/backward pass:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_detection_model_smoke_test.py --device cpu
```

## Step 12: Train Faster R-CNN Detection Baseline

Start with a small CUDA smoke training run:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_train_detection_frcnn.py --device cuda --epochs 1 --batch-size 1 --max-train-images 32 --max-test-images 16 --output-dir jun_model\demo1\models\detection_frcnn_smoke_cuda
```

Then run full Aperio holdout training:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_train_detection_frcnn.py --device cuda --epochs 5 --batch-size 2 --output-dir jun_model\demo1\models\detection_frcnn_aperio
```

With color/stain augmentation:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_train_detection_frcnn.py --device cuda --epochs 5 --batch-size 2 --color-jitter --output-dir jun_model\demo1\models\detection_frcnn_aperio_color_jitter
```

The script writes:

```text
config.json
training_log.csv
training_log.json
checkpoint_epoch_N.pt
latest.pt
final.pt
```

CPU training is very slow for Faster R-CNN. Use CUDA if available.

## Step 13: Evaluate Detection Baseline

Evaluate the trained detector:

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_evaluate_detection_frcnn.py --device cuda --batch-size 4 --checkpoint jun_model\demo1\models\detection_frcnn_aperio_b4\final.pt --coco jun_model\demo1\detection_tiles\Aperio_CS2\test_tiles_coco.json --output-dir jun_model\demo1\models\detection_frcnn_aperio_b4\eval
```

Current Aperio holdout detection result, IoU threshold `0.3`:

```text
Score threshold   Precision   Recall   F1
0.05              0.057       0.884    0.108
0.10              0.099       0.817    0.176
0.20              0.177       0.712    0.284
0.30              0.260       0.645    0.371
0.50              0.423       0.516    0.465
0.70              0.602       0.389    0.473
```

Best tested threshold:

```text
score threshold: 0.70
precision: 0.602
recall: 0.389
F1: 0.473
TP: 361
FP: 239
FN: 567
```

Color-jitter detector evaluation:

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_evaluate_detection_frcnn.py --device cuda --batch-size 4 --checkpoint jun_model\demo1\models\detection_frcnn_aperio_color_jitter\final.pt --coco jun_model\demo1\detection_tiles\Aperio_CS2\test_tiles_coco.json --output-dir jun_model\demo1\models\detection_frcnn_aperio_color_jitter\eval
```

Detection comparison:

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\compare_detection_results.py
```

Current best results, IoU threshold `0.3`:

```text
Model                       Score   Precision   Recall   F1
Faster R-CNN                0.70    0.602       0.389    0.473
Faster R-CNN + color jitter 0.70    0.604       0.727    0.660
```

Color jitter substantially improves detection recall and F1 on the held-out Aperio CS2 scanner.
