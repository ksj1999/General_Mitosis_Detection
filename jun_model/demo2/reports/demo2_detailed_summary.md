# Demo 2 Detailed Project Summary

## Project Goal

Demo 2 investigated whether pathology foundation models can classify mitotic versus non-mitotic candidates across scanner domains, and whether stain augmentation, scanner-adversarial learning, and parameter-efficient encoder adaptation improve source-domain robustness under scanner-aware validation.

This phase moved beyond the earlier pixel and detector experiments by using pretrained pathology encoders and a scanner-aware validation protocol. The final task remained candidate-level classification, not yet exhaustive whole-slide mitosis detection.

## Dataset and Labels

The experiment used the breast-cancer portion of MIDOG++:

- 150 pathology images: 50 per scanner.
- 4,435 annotated candidates.
- 1,721 mitotic figures (`label = 1`).
- 2,714 annotated non-mitotic candidates (`label = 0`).

| Scanner | Images | Candidates | Mitoses | Non-mitoses |
|---|---:|---:|---:|---:|
| Aperio CS2 | 50 | 1,612 | 688 | 924 |
| Hamamatsu S360 | 50 | 1,648 | 582 | 1,066 |
| Hamamatsu XR | 50 | 1,175 | 451 | 724 |

The source labels contain candidate bounding boxes and categories. The experiments used each box center to extract a pathology patch and used the category to define the binary target.

## Data Preparation

The preparation pipeline:

1. Audited image, annotation, category, scanner, split, and resolution metadata.
2. Read TIFF resolution tags and converted them into microns per pixel (MPP).
3. Generated candidate manifests containing coordinates, labels, scanner identity, crop geometry, edge flags, and ambiguity flags.
4. Extracted physically calibrated patches rather than using the same source-pixel crop for every scanner.
5. Padded unavailable edge pixels with white without shifting the candidate center.
6. Flagged negative-centered crops containing another annotated mitosis as ambiguous negatives.
7. Created image-level fold assignments and inherited those assignments at candidate level to prevent image leakage.

Two views were generated as `224 x 224` model inputs:

| View | Target resolution | Physical field of view |
|---|---:|---:|
| 20x equivalent | 0.50 um/px | 112 x 112 um |
| 40x equivalent | 0.25 um/px | 56 x 56 um |

Visual previews and validation experiments supported using the 40x representation. For Virchow2, 20x alone produced F1 `0.6315` and PR-AUC `0.6335`; direct 20x+40x concatenation produced F1 `0.7194` and PR-AUC `0.7335`, both below the 40x frozen probe (`0.7384` and `0.7769`).

## Validation Protocol

Three scanner folds were constructed. In each fold:

- One scanner's 50 images formed a locked held-out test domain.
- Source `train` images from the other two scanners formed the training set.
- Source `test` images from those two scanners formed the validation set.
- Train, validation, and test roles were image-disjoint.
- Model settings, checkpoints, and thresholds were selected using source-scanner validation only.

The metadata did not include patient identifiers. The protocol is slide-disjoint, but patient-disjoint status cannot be claimed.

The original frozen configurations received a one-time locked evaluation. Those test results were subsequently visible, so later post-hoc LoRA development was restricted to source validation. A fresh external or newly reserved dataset is required for an unbiased final claim.

## Foundation Models

Three pathology encoders were compared using each model's official preprocessing and appropriate pooling:

| Encoder | Approximate parameters | Feature representation |
|---|---:|---:|
| Phikon-v2 | 307M | Official 1,024-dimensional pooled output |
| Hibou-B | 86M | Official 768-dimensional pooled output or CLS fallback |
| Virchow2 | 632M | CLS plus mean patch token, 2,560 dimensions |

Hibou-B and Virchow2 required Hugging Face gated-model access. Weights were cached locally but were not redistributed.

## Frozen Linear-Probe Baselines

Frozen embeddings were cached once and classified with regularized logistic regression. Regularization and confidence thresholds were selected per fold using source validation.

| Frozen encoder | Validation F1 | Validation PR-AUC |
|---|---:|---:|
| Phikon-v2 | 0.6726 | 0.6600 |
| Hibou-B | 0.6537 | 0.6793 |
| Virchow2 | **0.7384** | **0.7769** |

Virchow2 was the strongest frozen representation. Hibou-B showed substantial scanner instability.

The one-time original locked-test results were:

| Frozen encoder | Test precision | Test recall | Test F1 (95% CI) | Test PR-AUC (95% CI) |
|---|---:|---:|---:|---:|
| Phikon-v2 | 0.550 | 0.739 | 0.595 (0.551-0.628) | 0.638 (0.603-0.672) |
| Hibou-B | 0.561 | 0.540 | 0.479 (0.446-0.506) | 0.593 (0.554-0.629) |
| Virchow2 | **0.621** | **0.837** | **0.707 (0.677-0.731)** | **0.764 (0.737-0.791)** |

These locked results were not reused to select the later LoRA configurations.

## Frozen-Feature Domain Experiments

### HED Stain Augmentation

The teammate's augmentation method was incorporated from `aaron_model/scripts/train_patch.py`:

- Convert RGB into HED stain space.
- Apply random multiplicative scale and additive bias to stain channels.
- Convert back to RGB and clamp to a valid range.
- Use `sigma = 0.05`, mode `both` for the main comparison.

For frozen probes, four deterministic augmented variants were embedded for each training candidate. Validation remained unaugmented. On Virchow2, augmentation changed the precision-recall operating point but left F1 and PR-AUC essentially unchanged.

### Scanner-Adversarial GRL

A neural probe was trained on frozen embeddings with:

- A 256-unit shared representation.
- A binary mitosis head.
- A scanner-classification head connected through gradient reversal.
- GRL strengths `0`, `0.01`, `0.1`, and `0.5`.
- Seeds 0, 1, and 2.

GRL did not outperform its architecture-matched neural baseline consistently. Stronger GRL mainly traded precision for recall and reduced PR-AUC.

### Frozen Stain + GRL

Combining cached HED variants with GRL produced only small, encoder-dependent changes. Phikon-v2 showed a hypothesis-generating gain, while Hibou-B and Virchow2 did not improve. This suggested that frozen embeddings limited how much augmentation and domain-adversarial objectives could reshape the representation.

## LoRA Encoder Adaptation

LoRA was introduced so the pathology representation itself could adapt without fully fine-tuning hundreds of millions of parameters.

Main settings:

- Rank 8, alpha 16, LoRA dropout 0.05.
- Phikon-v2 and Hibou-B: attention `query` and `value` projections.
- Virchow2: fused attention `qkv` projections.
- AdamW with learning rate and weight decay `1e-4`.
- Class-weighted binary cross-entropy.
- Mixed-precision training.
- Up to 8 epochs with patience-2 early stopping on validation PR-AUC.
- Confidence threshold selected on source validation.
- Effective batch size 8.
- Virchow2 used physical batch size 2 with four-step gradient accumulation on an RTX 3090.

The LoRA-only trainable parameter counts, including the classifier, were approximately:

- Phikon-v2: 787,457 of 304.1M parameters.
- Hibou-B: 295,681 of 86.0M parameters.
- Virchow2: 1,313,281 of 632.6M parameters.

Online LoRA stain augmentation perturbed every training batch before official model preprocessing; validation remained plain. LoRA-GRL attached a balanced scanner head to adapted features using `lambda = 0.01`. The combined model used both mechanisms.

## Cross-Model Seed-0 LoRA Ablation

| Encoder | Method | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| Phikon-v2 | LoRA | 0.7771 | 0.8111 | 0.7901 | 0.8408 |
| Phikon-v2 | LoRA + stain | 0.7789 | 0.8102 | 0.7925 | 0.8391 |
| Phikon-v2 | LoRA + GRL | 0.7711 | 0.8132 | 0.7910 | 0.8380 |
| Phikon-v2 | Combined | 0.7783 | 0.8270 | **0.8017** | **0.8459** |
| Hibou-B | LoRA | 0.7108 | **0.8521** | 0.7748 | 0.8452 |
| Hibou-B | LoRA + stain | 0.7241 | 0.8314 | 0.7731 | **0.8515** |
| Hibou-B | LoRA + GRL | 0.7299 | 0.8398 | **0.7802** | 0.8487 |
| Hibou-B | Combined | **0.7433** | 0.8212 | 0.7767 | 0.8491 |
| Virchow2 | LoRA | 0.7959 | 0.8315 | 0.8127 | 0.8745 |
| Virchow2 | LoRA + stain | 0.8136 | 0.8227 | 0.8178 | 0.8845 |
| Virchow2 | LoRA + GRL | **0.8216** | 0.8363 | 0.8285 | 0.8782 |
| Virchow2 | Combined | 0.8092 | **0.8549** | **0.8309** | **0.8872** |

LoRA substantially improved all three models over their frozen linear probes. Virchow2 remained the strongest encoder, and its combined method was the strongest seed-0 configuration. Hibou-B did not show the same synergy: GRL gave its highest F1, while stain augmentation gave its highest PR-AUC.

## Three-Seed Virchow2 Confirmation

The selected Virchow2 comparison was repeated with seeds 1 and 2 without changing hyperparameters.

| Seed | LoRA F1 | Combined F1 | Delta F1 | LoRA PR-AUC | Combined PR-AUC | Delta PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8127 | 0.8309 | +0.0182 | 0.8745 | 0.8872 | +0.0127 |
| 1 | 0.8134 | 0.8225 | +0.0091 | 0.8758 | 0.8841 | +0.0083 |
| 2 | **0.8190** | 0.8154 | -0.0037 | 0.8792 | **0.8814** | +0.0022 |
| **Mean (seed SD)** | **0.8151 (0.0028)** | **0.8229 (0.0064)** | **+0.0079** | **0.8765 (0.0020)** | **0.8842 (0.0024)** | **+0.0077** |

The combined method improved PR-AUC in all three seeds and F1 in two of three. Across nine seed-fold pairs, it improved F1 in seven and PR-AUC in eight. Mean paired improvements were positive on all three scanner folds.

At this stage, source validation suggested that stain augmentation plus weak scanner-adversarial learning might improve Virchow2 ranking quality after LoRA adaptation. The later post-hoc held-out evaluation was required to determine whether that validation gain transferred to the excluded scanner.

## Engineering Work Completed

The project now includes:

- Reproducible manifest and patch preparation scripts.
- Resolution-aware 20x and 40x extraction.
- Scanner-fold generation and leakage tests.
- Foundation-model registry and shared official preprocessing/pooling loader.
- Local gated-model cache handling.
- Frozen embedding caching and linear-probe evaluation.
- HED stain augmentation shared with the teammate workflow.
- Scanner-adversarial neural probes.
- Multi-scale cache alignment and fusion.
- LoRA training for Transformers and timm backends.
- Gradient accumulation for Virchow2 memory constraints.
- Fold-specific adapter checkpoint saving and fold resumption.
- Batch orchestration with timestamped logs and resumable experiment-level execution.
- Aggregate CSV and Markdown results.
- Thirteen passing data, pooling, augmentation, fold, fusion, scanner-probe, and GRL tests.

## Interpretation

1. Pathology foundation representations are much stronger than the initial simple patch models, but encoder choice matters.
2. Virchow2 is consistently the strongest frozen and adapted encoder.
3. Frozen-feature augmentation and GRL have limited impact because they cannot reshape the foundation representation.
4. LoRA provides the largest improvement while updating a small fraction of model parameters.
5. The source-validation advantage of stain augmentation plus weak GRL does not translate into a meaningful post-hoc held-out F1 improvement over LoRA alone.
6. Weak GRL does not remove scanner identity from the representation; a fresh linear probe recovers the source scanner almost perfectly.
7. Scanner information can remain strongly encoded even when held-out candidate classification is reasonably strong.
8. Per-scanner metrics and a fresh external confirmation remain necessary.

## Limitations

- LoRA was developed using source-scanner validation; its excluded-scanner evaluation is post-hoc rather than a fresh external test.
- The original locked tests were viewed before post-hoc LoRA development and cannot support a new unbiased claim.
- Patient identifiers were unavailable, so only slide-level disjointness was verified.
- Thresholds were selected independently within validation folds; a final deployed model needs one global cross-validated threshold.
- The current task classifies annotated candidates. It does not measure exhaustive full-image detection or false positives per whole slide.
- Three seeds provide an initial stability check but limited statistical power.
- Model access and licenses constrain redistribution, especially for Virchow2.

## Corrective Evaluation Protocol

Two analyses were added after protocol review.

First, `evaluate_lora_heldout.py` loads the exact source-validation-selected LoRA checkpoint and threshold for each fold, evaluates once on that fold's excluded scanner, saves candidate-level predictions, and computes image-level bootstrap intervals. No held-out labels are used for checkpoint or threshold selection. Because the original frozen held-out results were already viewed before LoRA development, these outputs must be reported as **post-hoc held-out-scanner evaluation**, not as an untouched confirmatory test. The valid comparison is LoRA-only versus combined LoRA under this same post-hoc protocol; the source-validation F1 `0.8309` must not be compared directly with the frozen held-out F1 `0.707`.

Second, `evaluate_scanner_probe.py` measures scanner information with a new classifier that was not used during GRL training. For every fold and seed, the adapted encoder is frozen, plain embeddings are extracted, a logistic-regression scanner probe is trained on source-train images, and it is evaluated on image-disjoint source-validation images. Balanced accuracy, macro F1, and cross-entropy are reported overall and separately among mitotic and non-mitotic candidates. Balanced accuracy near `0.50` means the two source scanners are less linearly distinguishable. It does not establish invariance to the unseen third scanner.

Commands and output locations are documented in `jun_model/demo2/README.md`. A genuinely fresh external scanner or institution remains the required final confirmation.

## Post-Hoc Held-Out LoRA Results

The selected checkpoints and source-validation thresholds were evaluated without retraining on each fold's excluded scanner. Results below are means across the three seeds, with each seed first macro-averaged across the three scanner folds.

| Method | Precision | Recall | F1, mean (seed SD) | PR-AUC, mean (seed SD) |
|---|---:|---:|---:|---:|
| LoRA | 0.7712 | **0.8532** | 0.8062 (0.0072) | 0.8765 (0.0048) |
| LoRA + stain + GRL | **0.7839** | 0.8349 | **0.8073 (0.0032)** | **0.8792 (0.0020)** |
| Combined minus LoRA | +0.0127 | -0.0183 | +0.0010 | +0.0027 |

The combined model raises precision while lowering recall. Its net changes of `+0.0010` F1 and `+0.0027` PR-AUC are too small to support a meaningful held-out improvement claim. It improves F1 in only three of nine paired seed-scanner comparisons and PR-AUC in five of nine, although its F1 is less variable across seeds.

| Held-out scanner | Combined minus LoRA F1 | Combined minus LoRA PR-AUC |
|---|---:|---:|
| Aperio CS2 | +0.0115 | -0.0002 |
| Hamamatsu S360 | -0.0048 | +0.0079 |
| Hamamatsu XR | -0.0036 | +0.0004 |

LoRA itself transfers reasonably well: its source-validation F1 of `0.8151` becomes `0.8062` in the post-hoc held-out analysis. The combined model falls from `0.8229` validation F1 to `0.8073` held-out F1, indicating that most of its apparent validation advantage does not transfer. Both LoRA results are descriptively above the original frozen Virchow2 held-out F1 of `0.707`, but this is not an unbiased confirmatory comparison because the original held-out results were visible before LoRA development.

## Scanner-Probe Results

A new logistic-regression probe was trained independently of the GRL head. It used source-train embeddings and image-disjoint source-validation embeddings. The probe was also fitted separately within each pathology class to prevent mitosis prevalence from acting as a scanner shortcut.

| Representation | All candidates | Non-mitotic only | Mitotic only |
|---|---:|---:|---:|
| Frozen Virchow2 | 1.0000 | 1.0000 | 1.0000 |
| LoRA | 1.0000 | 1.0000 | 0.9995 |
| LoRA + stain + GRL | 1.0000 | 1.0000 | 1.0000 |
| Two-scanner chance level | 0.5000 | 0.5000 | 0.5000 |

Values are balanced scanner-classification accuracy. Cross-entropy was also near zero for every representation, showing highly confident scanner predictions. Therefore, weak GRL (`lambda = 0.01`) did not produce scanner-invariant embeddings. This remains true within mitotic and non-mitotic candidates, so the result cannot be explained by differences in class prevalence.

Likely contributors are the weak adversarial coefficient, the restricted rank-8 LoRA update, and scanner information already being deeply encoded by the mostly frozen Virchow2 backbone. The GRL objective also sees only the two source scanners in each fold and cannot guarantee invariance to the third. Complete scanner invariance may not be necessary for useful classification, but it cannot be claimed from this experiment.

## Recommended Next Stage

1. Use plain Virchow2 LoRA as the current primary model; retain stain plus GRL as an ablation rather than the selected model.
2. Pool out-of-fold predictions and select one scanner-agnostic global threshold.
3. Produce scanner-stratified false-positive and false-negative galleries.
4. Integrate the classifier after a candidate proposal or detector stage.
5. Evaluate coordinate-matched detection precision, recall, F1, and FROC.
6. Test the locked pipeline on genuinely fresh external or newly reserved pathology data.

The current primary research model is Virchow2 at 40x with rank-8 `qkv` LoRA. Online HED stain augmentation (`sigma = 0.05`) plus weak scanner-adversarial GRL (`lambda = 0.01`) remains a documented ablation: it is slightly more seed-stable but does not materially improve post-hoc held-out performance or reduce scanner predictability.
