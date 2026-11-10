# Demo 1 Report: Cross-Scanner Robustness for Mitosis Detection

## Question

Does a model trained on some whole-slide scanners still work on a scanner it has never
seen, and does color/stain augmentation help?

## Data

MIDOG++ human breast cancer subset: 150 slides from 3 scanners.

```text
Aperio CS2       50 slides
Hamamatsu S360   50 slides
Hamammatsu XR    50 slides
```

Tumor type is held fixed at human breast cancer, so scanner is the only domain shift
being tested. Each experiment trains on two scanners and tests on the third
(leave-one-scanner-out).

## What I Did

**Part 1 — Patch classification.** Cropped 128x128 patches centered on each annotated
candidate and trained classifiers to separate mitotic from non-mitotic candidates:

1. Summary color/intensity features + logistic regression
2. Downsampled 32x32 raw pixels + logistic regression
3. Same, plus generic brightness/contrast/RGB augmentation
4. Same, plus augmentation guided by measured per-scanner RGB means
5. All of the above across all three scanner holdouts

**Part 2 — Full detection.** Moved from patch classification to actual detection:

6. Built COCO splits using only mitotic boxes as targets
7. Cut 512x512 tiles with balanced positive/negative sampling (2078 train, 1376 test)
8. Trained Faster R-CNN (ResNet50-FPN) on the Aperio CS2 holdout, with and without
   color jitter
9. Evaluated precision/recall/F1 at IoU 0.3 across a score-threshold sweep

## Results

### Patch classification (mean across 3 scanner holdouts)

```text
Model                      Mean F1   Mean Recall
Pixel baseline             0.429     0.508
Pixel + generic aug        0.463     0.564
Pixel + scanner-stat aug   0.442     0.535
```

Augmentation helped on two of three holdouts and was flat on Hamammatsu XR. Generic
augmentation beat the scanner-statistics version, so the RGB-mean shift was too crude.

**Important caveat:** all of these are below trivial baselines. Always predicting
"mitotic" scores F1 0.52-0.60 depending on the fold, and always predicting
"not mitotic" scores higher accuracy than any model. Flattened-pixel logistic
regression does not transfer across scanners. This is a real negative result, not a
win, and the augmentation gain is movement inside that failure regime.

### Detection, held-out Aperio CS2 (IoU 0.3, score threshold 0.70)

```text
Model                         Precision   Recall    F1
Faster R-CNN                  0.602       0.389     0.473
Faster R-CNN + color jitter   0.604       0.727     0.660
```

Color jitter roughly doubled recall at matched precision. Held-out loss agrees
independently: 0.0425 with jitter vs 0.0575 without.

## Conclusion

The patch-level experiments show that a weak pixel-level model fails to cross the
scanner gap entirely. Once the task was framed as real detection with a proper
backbone architecture, color/stain augmentation produced a large recall gain on an
unseen scanner, which supports the original hypothesis that cross-scanner robustness
is partly a color/stain problem.

## Limitations

The detection result is promising but not yet demonstrated:

- **Single run per arm, no random seed.** The two detectors differ in weight
  initialization and data order as well as augmentation, so the gap is not yet
  separable from seed variance. Needs 3+ seeds per arm.
- **Trained from random init.** No ImageNet pretrained weights were used, which is why
  absolute F1 sits well below published MIDOG baselines.
- **Threshold chosen on the test set**, and 0.70 was the highest value swept. The
  jitter model's F1 was still rising there, so its true optimum was never measured.
- **Recall denominator inflated 1.35x.** Overlapping tiles duplicate 180 of the 688
  test mitoses, so reported recall is per-tile-instance rather than per-mitosis.
- **One of three holdouts.** Detection was only run on Aperio CS2, the fold where
  augmentation looked best at patch level.
