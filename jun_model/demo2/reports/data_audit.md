# Demo 2 Data Audit

## Status

The candidate manifest, physical-scale crop geometry, and leave-one-scanner-out folds are prepared and validated. No foundation-model training has been started.

## Source Data

| Item | Count |
|---|---:|
| Pathology images | 150 |
| Images per scanner | 50 |
| Mitotic candidates | 1,721 |
| Non-mitotic candidates | 2,714 |
| Total candidates | 4,435 |

The three scanner domains are Aperio CS2, Hamamatsu S360, and `Hamammatsu XR` (spelled this way in the source metadata). All images originate from UMC Utrecht, so this study measures scanner-domain generalization within one source institution rather than cross-hospital generalization.

## Physical Resolution

MPP was derived per image from TIFF `XResolution`, `YResolution`, and `ResolutionUnit` tags.

| Scanner | Images | MPP range | Mean MPP |
|---|---:|---:|---:|
| Aperio CS2 | 50 | 0.253301 | 0.253301 |
| Hamamatsu S360 | 50 | 0.229848 | 0.229848 |
| Hamammatsu XR | 50 | 0.226317-0.227444 | 0.226759 |

The TIFF metadata supports MPP-aware extraction. MPP should remain an image-level manifest field rather than a hard-coded scanner constant.

## Scale Definitions

| Name | Model input | Target MPP | Field of view |
|---|---:|---:|---:|
| 20x | 224 x 224 | 0.50 um/px | 112 x 112 um |
| 40x | 224 x 224 | 0.25 um/px | 56 x 56 um |

Representative source crop widths are approximately 442-495 pixels at 20x and 221-247 pixels at 40x. The crop is centered before any padding, so edge handling does not shift the annotated candidate away from the model-input center.

## Candidate Quality Checks

| Check | 20x | 40x |
|---|---:|---:|
| Crops touching an image edge | 572 | 261 |
| Negative-centered crops containing another mitosis | 346 | 73 |

Additional findings:

- 48 source 50 x 50 annotation boxes cross an image border by at most 18 pixels. Their centers remain inside the source image. They are retained, flagged, and padded during extraction.
- No exact duplicate annotation groups were found.
- No invalid zero-area boxes were found.
- All annotation centers and image dimensions use the same coordinate system in the inspected data.

The large difference in ambiguous-negative counts means patch scale can materially change the classification task. Scale and ambiguity policy must be selected using training-scanner validation data, not the held-out scanner.

## LOSO Fold Counts

| Held-out scanner | Role | Images | Candidates | Mitotic | Non-mitotic |
|---|---|---:|---:|---:|---:|
| Aperio CS2 | train | 78 | 2,255 | 834 | 1,421 |
| Aperio CS2 | validation | 22 | 568 | 199 | 369 |
| Aperio CS2 | test | 50 | 1,612 | 688 | 924 |
| Hamamatsu S360 | train | 78 | 2,210 | 915 | 1,295 |
| Hamamatsu S360 | validation | 22 | 577 | 224 | 353 |
| Hamamatsu S360 | test | 50 | 1,648 | 582 | 1,066 |
| Hamammatsu XR | train | 78 | 2,579 | 1,015 | 1,564 |
| Hamammatsu XR | validation | 22 | 681 | 255 | 426 |
| Hamammatsu XR | test | 50 | 1,175 | 451 | 724 |

Within each fold, the two training scanners contribute their 39 source-training images to training and their 11 source-test images to validation. All 50 images from the held-out scanner remain locked for final testing. Automated checks confirmed no image overlap among roles.

## Known Limitations

- Patient identifiers are not available in the prepared metadata. The folds are slide-disjoint, but patient-disjoint status cannot be verified.
- The source images are large pathology regions, not complete whole-slide images.
- TIFF resolution tags are internally consistent, but no independent scanner calibration file is available.
- Candidate labels describe the centered candidate. Larger crops can contain additional labeled objects.
- Visual review confirms extraction geometry, but it is not evidence for selecting 20x or 40x.

## Environment Audit

The CUDA environment contains PyTorch `2.5.1+cu121`, torchvision `0.20.1+cu121`, Transformers `4.46.3`, scikit-learn `1.5.2`, timm `1.0.11`, PEFT `0.13.2`, and Accelerate `1.1.1`. Phikon-v2 was downloaded into a Git-ignored local cache and successfully evaluated on the RTX 3090. The selected encoder set is Phikon-v2, Hibou-B, and Virchow2. Hibou-B and Virchow2 remain unavailable locally until their Hugging Face terms are accepted and this machine is authenticated.

Before installation or model download, verify the official checkpoint identifier, license/access requirements, image normalization, input size, and LoRA-compatible layers for Phikon-v2, UNI, and Virchow2.

## Completed Scale Decision

Run a training-scanner-only 20x versus 40x linear-probe pilot. For each LOSO fold:

1. Fit on the 78 training images from the two source scanners.
2. Select regularization and the classification threshold using only the 22 validation images.
3. Compare scale using aggregated source-validation performance.
4. Lock one scale before evaluating any held-out-scanner test candidates.

This decision was completed with the Phikon-v2 validation-only pilot below. The resulting 40x configuration was written to `jun_model/demo2/configs/phikon_v2_linear_probe_v1.json` before held-out testing.

## Phikon-v2 Scale Pilot

Phikon-v2 frozen embeddings were extracted for all candidates at both physical scales. Each cache contains 4,435 candidate rows and 1,024-dimensional pooled features. Logistic-regression regularization and confidence thresholds were selected independently in each fold using only source-scanner validation images.

| Scale | Mean validation precision | Mean validation recall | Mean validation F1 | Mean validation PR-AUC |
|---|---:|---:|---:|---:|
| 20x | 0.449 | 0.839 | 0.583 | 0.547 |
| 40x | 0.585 | 0.793 | 0.673 | 0.660 |

The 40x representation improved validation F1 in all three folds and increased mean validation F1 by 0.089. It also increased mean PR-AUC by 0.113, so the result is not explained only by threshold selection. The 40x scale is locked for the first Phikon-v2 linear-probe test. Held-out test labels were not evaluated during this selection.

## Locked Phikon-v2 Test

After locking scale, regularization, and fold-specific thresholds, the 40x linear probe was evaluated once on each held-out scanner.

| Held-out scanner | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aperio CS2 | 0.708 | 0.461 | 0.558 | 0.657 | 317 | 131 | 371 |
| Hamamatsu S360 | 0.458 | 0.907 | 0.609 | 0.646 | 528 | 624 | 54 |
| Hamammatsu XR | 0.485 | 0.849 | 0.618 | 0.611 | 383 | 406 | 68 |
| Macro mean | 0.550 | 0.739 | 0.595 | 0.638 | - | - | - |

PR-AUC is relatively stable across scanners, but precision and recall shift substantially. The validation-selected decision thresholds are conservative on Aperio and permissive on S360 and XR. This indicates scanner-dependent probability calibration even when ranking quality transfers reasonably well. Test thresholds must remain unchanged; calibration is a future training-scanner-only experiment.

Image-level bootstrap intervals with 2,000 resamples were used to account for candidates clustered within the same pathology image:

| Macro metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| Precision | 0.550 | 0.518-0.579 |
| Recall | 0.739 | 0.689-0.782 |
| F1 | 0.595 | 0.551-0.628 |
| PR-AUC | 0.638 | 0.603-0.672 |

## Next Step

Implement a Phikon-v2 LoRA baseline at the locked 40x scale. Validate that gradients reach only the intended adapter and classification-head parameters, record peak GPU memory, and run one source-validation smoke fold before launching the three-fold baseline. Stain augmentation and GRL should be added only after the plain LoRA control is stable.
