# Next Experiments

The current Demo 1 result shows that generic stain/color augmentation improves mean leave-one-scanner-out patch performance.

Current mean held-out result:

```text
pixel_baseline:   F1 0.429, recall 0.508
pixel_augmented:  F1 0.463, recall 0.564
```

## 1. Scanner-Statistics Augmentation

Goal: replace generic RGB jitter with augmentation guided by the observed scanner color profiles.

Current scanner means:

```text
Aperio CS2:      RGB mean (0.800, 0.597, 0.693)
Hamamatsu S360:  RGB mean (0.812, 0.574, 0.741)
Hamammatsu XR:   RGB mean (0.775, 0.566, 0.794)
```

Experiment run:

```text
For each training patch:
  1. Estimate its RGB mean.
  2. Pick a scanner-style RGB target from training scanners only.
  3. Shift the patch color partway toward that target.
  4. Add mild brightness/contrast jitter.
```

Result:

```text
Held-out scanner | Baseline F1 | Generic Aug F1 | Scanner-Stat F1
Aperio CS2       | 0.412       | 0.450          | 0.415
Hamamatsu S360   | 0.346       | 0.413          | 0.384
Hamammatsu XR    | 0.529       | 0.525          | 0.529
Mean             | 0.429       | 0.463          | 0.442
```

Scanner-statistics augmentation improved over baseline on average, but did not beat generic augmentation. The simple RGB-mean shift is likely too crude; if we revisit this, use a more realistic stain normalization or H&E stain-vector method rather than only scanner RGB means.

## 2. Threshold Calibration

Goal: see whether scanner shift is partly a probability calibration issue.

Experiment:

```text
Use training scanner data to choose a decision threshold.
Apply the same threshold to the held-out scanner.
Compare F1, precision, and recall against the fixed 0.5 threshold.
```

This is especially relevant because `Hamammatsu XR` had high recall but low precision, suggesting over-calling mitosis.

## 3. Stronger Patch Model

Goal: move beyond flattened pixel logistic regression.

Possible models:

```text
small CNN
pretrained ResNet feature extractor
linear classifier on pretrained embeddings
```

The current CNN script is slow on this machine, so the practical next move is either tensor caching plus a lighter architecture, or extracting features once and training a linear classifier.

## 4. Full Detection

Goal: return from patch classification to the actual detection framing.

Patch classification answers:

```text
Can a model distinguish mitotic from non-mitotic candidates across scanners?
```

Full detection asks:

```text
Can a model find mitotic figures in the image?
```

The patch task is a useful first robustness experiment, but the final project should describe it clearly as patch-level unless full detection is added later.
