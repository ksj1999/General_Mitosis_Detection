# Demo 1 Presentation: Scanner Generalization for Mitotic Figure Detection

## Slide 1: Title

**Scanner Generalization for Mitotic Figure Detection**

Demo 1, CSE 4109 AI for Health

Team project: robust mitotic figure detection on real breast cancer pathology images

Speaker note:
This demo shows real progress: a prepared pathology dataset, a working detection pipeline, scanner-domain experiments, and early robustness results.

## Slide 2: Project Motivation

Pathology AI models can fail when image appearance changes across scanners.

Our research question:

```text
Can a mitotic figure detector trained on some scanner domains generalize to an unseen scanner?
```

Why it matters:

- mitotic figures are clinically important for cancer grading
- scanner and staining differences change image color/contrast
- deployment needs robustness beyond one scanner source

Speaker note:
Frame this as a real-world generalization problem, not only a model accuracy problem.

## Slide 3: Dataset And Task

Dataset: MIDOG++ human breast cancer subset

| Item | Value |
| --- | ---: |
| TIFF images | 150 |
| Object annotations | 4435 |
| Tumor type | human breast cancer |
| Scanner domains | 3 |
| Task | mitotic figure detection |

Scanners:

- Aperio CS2
- Hamamatsu S360
- Hamammatsu XR

Speaker note:
Mention that we made a shareable subset with images, metadata, and COCO-style labels.

## Slide 4: Experimental Design

Evaluation design: leave-one-scanner-out.

```text
Train on two scanners.
Test on the held-out scanner.
```

This tests scanner generalization while keeping tumor type fixed.

Pipeline:

```text
MIDOG++ annotations
-> breast cancer subset
-> detection tiles
-> Faster R-CNN detector
-> held-out scanner evaluation
```

Speaker note:
LOSO is the core method. It makes the demo stronger because we are not randomly mixing scanner domains.

## Slide 5: What We Built For Demo 1

Completed:

- cleaned breast cancer subset for teammate use
- metadata and label README
- patch-level scanner robustness experiments
- 512 x 512 detection tile datasets
- Faster R-CNN training and evaluation scripts
- checkpoint sweep evaluation
- visual false positive / false negative panels
- preliminary stain normalization experiment

Speaker note:
This directly answers the project requirement to show something real.

## Slide 6: Initial Patch-Level LOSO Results

Logistic regression on cropped candidates: baseline vs color augmentation.

| Held-out scanner | Base P | Base R | Base F1 | Aug P | Aug R | Aug F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aperio CS2 | 0.444 | 0.384 | 0.412 | 0.460 | 0.440 | 0.450 | +0.038 |
| Hamamatsu S360 | 0.346 | 0.345 | 0.346 | 0.372 | 0.466 | 0.413 | +0.068 |
| Hamammatsu XR | 0.396 | 0.794 | 0.529 | 0.395 | 0.785 | 0.525 | -0.004 |
| **Mean** | **0.395** | **0.508** | **0.429** | **0.409** | **0.564** | **0.463** | **+0.034** |

Takeaway:

```text
Color augmentation improved mean F1 from 0.429 to 0.463 and mean recall from 0.508 to 0.564.
```

Speaker note:
This was a useful first signal, but patch classification is not the final clinical task.

## Slide 7: Full Detection Results

Full object detection used Faster R-CNN on real pathology tiles.

| Model | Mean precision | Mean recall | Mean F1 |
| --- | ---: | ---: | ---: |
| Stable baseline | 0.713 | 0.755 | 0.728 |
| Strong color jitter | 0.702 | 0.700 | 0.698 |

Takeaway:

```text
Strong color jitter did not improve full detection.
```

Speaker note:
This is important: a method that helped patches did not automatically improve detection.

## Slide 8: Normalization Result

Tested simple Reinhard-style RGB normalization on held-out Hamamatsu S360.

| Method | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stable baseline | 0.633 | 0.817 | 0.714 | 379 | 147 |
| Reinhard RGB normalization | 0.709 | 0.721 | 0.715 | 237 | 224 |

Takeaway:

```text
Normalization tied baseline F1 and reduced false positives.
```

Speaker note:
Be clear that normalization is promising but not yet proven across all scanners.

## Slide 9: Visual Evidence

We generated visual panels comparing model predictions against ground truth.

Color coding:

```text
green = ground truth
red = baseline prediction
blue = comparison model prediction
```

These panels help identify:

- false positives that look visually similar to mitoses
- false negatives caused by conservative predictions
- scanner/stain cases where the detector struggles

Speaker note:
This slide is where you show one actual JPG output. It makes the demo concrete.

## Slide 10: Interpretation And Next Steps

What we learned:

- scanner generalization is measurable with LOSO evaluation
- patch-level augmentation helped, but full detection behaved differently
- stable training mattered more than naive color jitter
- stain normalization is a better next direction than stronger jitter

Next:

- run normalization on all held-out scanners
- generate baseline vs normalization error panels
- try pathology-specific stain normalization such as Macenko
- later compare with domain-adversarial learning

Speaker note:
End with a clean research story: we built the pipeline, found a negative result for naive jitter, and identified a more promising next step.

