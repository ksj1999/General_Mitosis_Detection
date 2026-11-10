# Demo 2: Scanner-Generalized Mitosis Classification

Demo 2 evaluates pathology foundation models for scanner-aware candidate-level mitosis classification. Later LoRA experiments use source-scanner validation; their unseen-scanner analysis is explicitly post-hoc because the original held-out results had already been viewed.

## Current Stage

The data audit, frozen linear-probe validation, configuration locking, and one-time held-out evaluation for Phikon-v2, Hibou-B, and Virchow2 are complete. Do not use held-out scanner results to revise patch scale, checkpoints, thresholds, augmentation strength, or fine-tuning settings.

## Prepare The Manifests

From the repository root:

```powershell
.\.venv-det\Scripts\python.exe demo2\prepare_demo2_data.py
.\.venv-det\Scripts\python.exe demo2\preview_patch_scales.py
.\.venv-det\Scripts\python.exe demo2\extract_candidate_patches.py
```

Generated files:

- `data/images.csv`: one row per pathology image, including TIFF-derived MPP.
- `data/candidates.csv`: one row per annotated candidate, with coordinates, labels, 20x/40x crop geometry, edge flags, and ambiguity flags.
- `data/fold_images.csv`: image-level train/validation/test assignment for every held-out-scanner fold.
- `data/fold_candidates.csv`: candidate-level fold assignments inherited from the image assignments.
- `data/split_summary.csv`: counts by fold, role, scanner, and label.
- `data/audit.json`: machine-readable audit summary.
- `outputs/patch_scale_preview.jpg`: side-by-side 20x and 40x visual checks.
- `patches/20x` and `patches/40x`: generated model inputs, ignored by Git.

## Scale Definitions

Both scales produce a `224 x 224` model input:

| Scale | Target MPP | Physical field of view |
|---|---:|---:|
| 20x equivalent | 0.50 um/px | 112 x 112 um |
| 40x equivalent | 0.25 um/px | 56 x 56 um |

Source crop dimensions are calculated independently for every image from TIFF `XResolution`, `YResolution`, and `ResolutionUnit` tags. Crops are resized only after extracting the corresponding physical field of view.

## Fold Protocol

For each fold:

- All 50 images from one scanner form the locked test set.
- Images marked `train` in the source MIDOG++ split from the other two scanners form the training set.
- Images marked `test` in the source split from the other two scanners form the validation set.
- No image appears in more than one role within a fold.

The metadata does not expose patient identifiers. The protocol is slide-disjoint, but patient-disjoint status cannot be claimed without additional source metadata.

## Label Policy

- `label = 1`: mitotic figure.
- `label = 0`: annotated non-mitotic candidate.
- `ambiguous_negative_20x` and `ambiguous_negative_40x` flag negative-centered crops containing another annotated mitosis.
- `bbox_outside_image` flags source 50 x 50 boxes that slightly cross an image border. Their centers remain valid; extraction preserves the center and pads unavailable pixels with white.
- Ambiguous negatives must be handled using a policy chosen from training-scanner validation only. Recommended first comparison: retain all candidates versus exclude flagged negatives from training, while keeping validation and test reporting transparent.

## Planned Experiment Order

1. Visually inspect the generated scale preview and audit counts.
2. Select 20x or 40x using training-scanner validation only.
3. Run frozen linear probes for Phikon-v2, Hibou-B, and Virchow2.
4. Use Phikon-v2 for initial LoRA development, then confirm the strongest method on Hibou-B and Virchow2 as resources allow.
5. Compare LoRA baseline, stain augmentation, GRL, and their combination.
6. Integrate the strongest locked classifier into ROI-level detection.

The current `.venv-det` has CUDA-enabled PyTorch, Transformers, scikit-learn, timm, PEFT, and Accelerate. Model behavior is defined in `configs/foundation_models.json`; the shared loader preserves each encoder's official preprocessing and pooling rather than applying Phikon's settings to every model.

## Three-Model Study

| Key | Encoder | Parameters | Access | Frozen embedding |
|---|---|---:|---|---|
| `phikon_v2` | Phikon-v2 | ~307M | Public | Official pooled output, 1,024 dimensions |
| `hibou_b` | Hibou-B | ~86M | Accept Hugging Face terms | Official processor, pooled output or CLS fallback, 768 dimensions |
| `virchow2` | Virchow2 | ~632M | Institutional approval | CLS concatenated with mean patch token, 2,560 dimensions |

Model weights are not distributed with this repository. Each collaborator should accept the applicable Hugging Face terms and authenticate independently. Virchow2 specifically prohibits redistribution and is restricted to approved non-commercial academic research under its model terms; Hibou-B is listed under Apache-2.0 but still requires accepting its access conditions on Hugging Face.

Before downloading the gated models:

1. Accept access at `https://huggingface.co/histai/hibou-b`.
2. Request and accept access at `https://huggingface.co/paige-ai/Virchow2` using an institutional Hugging Face email.
3. Authenticate locally:

```powershell
.\.venv-det\Scripts\hf.exe auth login
.\.venv-det\Scripts\hf.exe auth whoami
```

Authentication and gated-model approval are machine- and user-specific; verify them before downloading weights.

## Phikon-v2 Frozen Embeddings

Install the pinned dependency, then cache each scale once:

```powershell
.\.venv-det\Scripts\python.exe -m pip install -r demo2\requirements-foundation.txt
.\.venv-det\Scripts\python.exe demo2\cache_foundation_embeddings.py --model-key phikon_v2 --scale 20x --output demo2\cache\phikon_v2_20x.npz
.\.venv-det\Scripts\python.exe demo2\cache_foundation_embeddings.py --model-key phikon_v2 --scale 40x --output demo2\cache\phikon_v2_40x.npz
```

The first embedding command downloads the public Phikon-v2 checkpoint into `demo2/.hf_cache`. Generated patches, weights, embeddings, and trained models are excluded from Git.

After authentication and approval, create the other locked-40x embedding caches:

```powershell
.\.venv-det\Scripts\python.exe demo2\cache_foundation_embeddings.py --model-key hibou_b --scale 40x --output demo2\cache\hibou_b_40x.npz
.\.venv-det\Scripts\python.exe demo2\cache_foundation_embeddings.py --model-key virchow2 --scale 40x --output demo2\cache\virchow2_40x.npz
```

Run `run_linear_probe_validation.py` on these caches first. Do not run their held-out test evaluations until the three model configurations are reviewed and locked.

Run validation-only linear probes for the scale pilot:

```powershell
.\.venv-det\Scripts\python.exe demo2\run_linear_probe_validation.py --features demo2\cache\phikon_v2_20x.npz --output-dir demo2\models\phikon_v2_20x_validation
.\.venv-det\Scripts\python.exe demo2\run_linear_probe_validation.py --features demo2\cache\phikon_v2_40x.npz --output-dir demo2\models\phikon_v2_40x_validation
```

These commands do not evaluate held-out test candidates. They select logistic-regression regularization and confidence thresholds using each fold's source-scanner validation images only.

After the scale and fold settings are written to a reviewed locked config, run the held-out test once:

```powershell
.\.venv-det\Scripts\python.exe demo2\evaluate_locked_linear_probe.py --features demo2\cache\phikon_v2_40x.npz --config demo2\configs\phikon_v2_linear_probe_v1.json --output-dir demo2\models\phikon_v2_40x_locked_test
.\.venv-det\Scripts\python.exe demo2\bootstrap_locked_metrics.py --predictions demo2\models\phikon_v2_40x_locked_test\locked_test_predictions.csv --output demo2\models\phikon_v2_40x_locked_test\bootstrap_metrics.json
```

The Hibou-B and Virchow2 configurations were locked on 2026-11-09 from source-scanner validation only. The following commands record the one-time evaluations that were run; do not rerun them for model or threshold selection, and do not change either v1 config afterward:

```powershell
.\.venv-det\Scripts\python.exe demo2\evaluate_locked_linear_probe.py --features demo2\cache\hibou_b_40x.npz --config demo2\configs\hibou_b_linear_probe_v1.json --output-dir demo2\models\hibou_b_40x_locked_test
.\.venv-det\Scripts\python.exe demo2\bootstrap_locked_metrics.py --predictions demo2\models\hibou_b_40x_locked_test\locked_test_predictions.csv --output demo2\models\hibou_b_40x_locked_test\bootstrap_metrics.json

.\.venv-det\Scripts\python.exe demo2\evaluate_locked_linear_probe.py --features demo2\cache\virchow2_40x.npz --config demo2\configs\virchow2_linear_probe_v1.json --output-dir demo2\models\virchow2_40x_locked_test
.\.venv-det\Scripts\python.exe demo2\bootstrap_locked_metrics.py --predictions demo2\models\virchow2_40x_locked_test\locked_test_predictions.csv --output demo2\models\virchow2_40x_locked_test\bootstrap_metrics.json
```

## Frozen Encoder Results

All values below are macro means over the three held-out-scanner folds. Confidence intervals use 2,000 image-level bootstrap samples. These tests have been evaluated and must not be rerun for model or threshold selection.

| Encoder | Validation F1 | Test precision | Test recall | Test F1 (95% CI) | Test PR-AUC (95% CI) |
|---|---:|---:|---:|---:|---:|
| Phikon-v2 | 0.673 | 0.550 | 0.739 | 0.595 (0.551-0.628) | 0.638 (0.603-0.672) |
| Hibou-B | 0.654 | 0.561 | 0.540 | 0.479 (0.446-0.506) | 0.593 (0.554-0.629) |
| Virchow2 | **0.738** | **0.621** | **0.837** | **0.707 (0.677-0.731)** | **0.764 (0.737-0.791)** |

Virchow2 is the strongest frozen encoder under the locked protocol. Hibou-B is unstable across scanners: its S360 held-out fold has 0.155 recall and 0.249 F1 despite 0.804 recall and 0.646 F1 on source validation. This is evidence of scanner-domain shift, not a reason to retune the locked Hibou configuration.

## Stain-Augmented Linear Probe

This post-hoc experiment uses Aaron's HED-space jitter from `aaron_model/scripts/train_patch.py` at teammate revision `30e9ee9bd947dfdda513264194dd25ec3d873e25`. Each RGB patch is converted to HED, each stain channel receives a random multiplicative scale and additive bias in the `sigma = 0.05` range, and the result is converted back to RGB. Four deterministic variants are cached for each candidate using seed 0, matching the teammate frozen-probe design.

Only training rows are expanded with augmented embeddings. Validation always uses the original unaugmented Virchow2 embeddings. Existing locked test outputs are not used for selection and must not be rerun to tune this method.

```powershell
.\.venv-det\Scripts\python.exe demo2\cache_stain_augmented_embeddings.py --model-key virchow2 --scale 40x --variants 4 --sigma 0.05 --stain-mode both --seed 0 --output demo2\cache\virchow2_40x_hed_seed0.npz --device cuda

.\.venv-det\Scripts\python.exe demo2\run_linear_probe_validation.py --features demo2\cache\virchow2_40x.npz --augmented-features demo2\cache\virchow2_40x_hed_seed0.npz --output-dir demo2\models\virchow2_40x_hed_seed0_validation
```

Lower-strength comparison:

```powershell
.\.venv-det\Scripts\python.exe demo2\cache_stain_augmented_embeddings.py --model-key virchow2 --scale 40x --variants 4 --sigma 0.025 --stain-mode both --seed 0 --output demo2\cache\virchow2_40x_hed_sigma0025_seed0.npz --device cuda

.\.venv-det\Scripts\python.exe demo2\run_linear_probe_validation.py --features demo2\cache\virchow2_40x.npz --augmented-features demo2\cache\virchow2_40x_hed_sigma0025_seed0.npz --output-dir demo2\models\virchow2_40x_hed_sigma0025_seed0_validation
```

Validation result:

| Virchow2 training | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Plain embeddings | 0.6943 | 0.7916 | **0.7384** | **0.7769** |
| HED jitter, 4 variants | 0.6859 | **0.8040** | 0.7376 | 0.7767 |
| HED jitter, 4 variants, `sigma=0.025` | 0.6802 | **0.8080** | 0.7364 | 0.7768 |

At `sigma=0.05`, HED jitter traded 0.0083 precision for 0.0125 recall, while mean F1 (-0.0007) and PR-AUC (-0.0002) were effectively unchanged. Reducing the strength to `sigma=0.025` increased recall by 0.0164 over plain training but reduced precision by 0.0141; F1 fell by 0.0020 and PR-AUC was effectively unchanged. Neither single-seed setting is promoted to locked testing. A useful follow-up is to test fewer augmented variants on source validation, then confirm any selected setting across multiple seeds before one final locked evaluation.

## Scanner-Adversarial Neural Probe

The GRL experiment uses frozen Virchow2 embeddings with a trainable `256`-unit MLP encoder, dropout `0.2`, a binary mitosis head, and a scanner head connected through gradient reversal. The `lambda=0` run is an architecture-matched neural baseline: it uses the same encoder and mitosis head but does not optimize scanner loss. Training uses AdamW, class-balanced losses, and early stopping on source-validation PR-AUC. Thresholds are also selected only on source validation.

The sweep evaluates GRL strengths `0`, `0.01`, `0.1`, and `0.5` across seeds `0`, `1`, and `2` and all three scanner folds:

```powershell
.\.venv-det\Scripts\python.exe demo2\run_adversarial_probe_validation.py --features demo2\cache\virchow2_40x.npz --output-dir demo2\models\virchow2_grl_validation --grl-strengths 0 0.01 0.1 0.5 --seeds 0 1 2 --hidden-dim 256 --dropout 0.2 --epochs 100 --patience 12 --device cuda
```

| Method | GRL strength | Precision | Recall | F1, mean (seed SD) | PR-AUC, mean (seed SD) |
|---|---:|---:|---:|---:|---:|
| Neural baseline | 0 | 0.6881 | 0.7863 | **0.7309 (0.0018)** | 0.7712 (0.0038) |
| Scanner-adversarial | 0.01 | **0.6906** | 0.7763 | 0.7293 (0.0047) | **0.7723 (0.0052)** |
| Scanner-adversarial | 0.1 | 0.6773 | 0.7903 | 0.7283 (0.0109) | 0.7677 (0.0024) |
| Scanner-adversarial | 0.5 | 0.6682 | **0.8058** | 0.7289 (0.0051) | 0.7630 (0.0031) |

No GRL strength improves mean F1 over the matched neural baseline. `lambda=0.01` raises mean PR-AUC by only 0.0011 while lowering F1 and increasing seed variation. Higher strengths produce a precision-to-recall tradeoff and progressively lower PR-AUC. The selected setting is therefore `lambda=0`, and the adversarial probe is not promoted to locked testing. The original logistic-regression Virchow2 probe also remains stronger than the neural baseline (F1 `0.7384`, PR-AUC `0.7769`).

### Combined Stain Augmentation and GRL at 40x

This ablation combines the original teammate HED jitter (`sigma=0.05`, four deterministic variants) with the scanner-adversarial neural probe. Only training candidates are expanded; source validation always uses plain 40x Virchow2 embeddings. Labels and scanner identities are repeated in the same order as the augmented feature variants.

```powershell
.\.venv-det\Scripts\python.exe demo2\run_adversarial_probe_validation.py --features demo2\cache\virchow2_40x.npz --augmented-features demo2\cache\virchow2_40x_hed_seed0.npz --output-dir demo2\models\virchow2_40x_hed_grl_validation --grl-strengths 0 0.01 0.1 0.5 --seeds 0 1 2 --hidden-dim 256 --dropout 0.2 --epochs 100 --patience 12 --device cuda
```

| 40x neural training | GRL strength | Precision | Recall | F1, mean (seed SD) | PR-AUC, mean (seed SD) |
|---|---:|---:|---:|---:|---:|
| Plain baseline | 0 | 0.6881 | 0.7863 | **0.7309 (0.0018)** | **0.7712 (0.0038)** |
| HED augmentation | 0 | 0.6698 | **0.7962** | 0.7268 (0.0030) | 0.7670 (0.0057) |
| HED augmentation + GRL | 0.01 | 0.6806 | 0.7792 | 0.7249 (0.0012) | 0.7674 (0.0048) |
| HED augmentation + GRL | 0.1 | **0.6898** | 0.7776 | 0.7303 (0.0027) | 0.7648 (0.0040) |
| HED augmentation + GRL | 0.5 | 0.6765 | 0.7898 | 0.7275 (0.0022) | 0.7670 (0.0054) |

Within augmented neural training, `lambda=0.1` recovers 0.0035 F1 over augmentation alone. However, it remains 0.0007 below the matched plain neural baseline and loses 0.0065 PR-AUC. It is also weaker than the original 40x logistic-regression probe. The combined method is therefore not promoted to locked testing.

### Cross-Encoder Stain and GRL Ablation

The same 40x neural protocol was repeated for Phikon-v2 and Hibou-B. Each comparison uses the same hidden dimension, optimization settings, four HED variants, three seeds, scanner folds, GRL strengths, and source-validation-only selection as Virchow2. Plain neural/GRL sweeps were also run for each encoder so the augmentation comparison does not mix classifier architectures.

```powershell
# Replace MODEL with phikon_v2 or hibou_b.
.\.venv-det\Scripts\python.exe demo2\cache_stain_augmented_embeddings.py --model-key MODEL --scale 40x --variants 4 --sigma 0.05 --stain-mode both --seed 0 --output demo2\cache\MODEL_40x_hed_seed0.npz --device cuda

.\.venv-det\Scripts\python.exe demo2\run_adversarial_probe_validation.py --features demo2\cache\MODEL_40x.npz --augmented-features demo2\cache\MODEL_40x_hed_seed0.npz --output-dir demo2\models\MODEL_40x_hed_grl_validation --grl-strengths 0 0.01 0.1 0.5 --seeds 0 1 2 --hidden-dim 256 --dropout 0.2 --epochs 100 --patience 12 --device cuda
```

| Encoder | Original linear F1 / PR-AUC | Best plain neural/GRL F1 / PR-AUC | HED neural F1 / PR-AUC | Best HED+GRL F1 / PR-AUC |
|---|---:|---:|---:|---:|
| Phikon-v2 | 0.6726 / 0.6600 | 0.6607 / 0.6633 (`lambda=0.5`) | 0.6702 / 0.6846 | **0.6741 / 0.6861** (`lambda=0.01`) |
| Hibou-B | **0.6537 / 0.6793** | 0.6507 / **0.6848** (`lambda=0.1`) | 0.6416 / 0.6695 | 0.6470 / 0.6827 (`lambda=0.5`) |
| Virchow2 | **0.7384 / 0.7769** | 0.7309 / 0.7712 (`lambda=0`) | 0.7268 / 0.7670 | 0.7303 / 0.7648 (`lambda=0.1`) |

Phikon-v2 is the only encoder where the combined method improves both mean validation F1 and PR-AUC over its original linear probe: +0.0015 F1 and +0.0261 PR-AUC. Most of the PR-AUC improvement already appears with HED augmentation, while weak GRL adds another 0.0039 F1. Hibou-B and Virchow2 do not improve under the combination. Because the locked test results were viewed before this post-hoc experiment, the Phikon result must be treated as hypothesis-generating and cannot be used for another nominally locked evaluation on the same test data.

## Phikon-v2 LoRA Pilot

The first encoder-adaptation experiment applies rank-8 LoRA adapters to the `query` and `value` projections in all Phikon-v2 attention blocks. The backbone remains frozen apart from 786,432 adapter parameters; including the linear mitosis head, 787,457 of 304,139,265 parameters are trainable (0.26%). This pilot deliberately excludes stain augmentation and GRL so its effect can be attributed to encoder adaptation.

Training uses class-weighted binary cross-entropy, AdamW, mixed precision, and early stopping on source-validation PR-AUC with patience 2. The best checkpoint and classification threshold are selected independently within each source-validation fold. No held-out test candidates are evaluated.

```powershell
.\.venv-det\Scripts\python.exe demo2\run_lora_probe_validation.py --model-key phikon_v2 --scale 40x --output-dir demo2\models\phikon_v2_40x_lora_validation --seeds 0 --epochs 8 --patience 2 --batch-size 8 --eval-batch-size 16 --offline
```

| Phikon-v2 method | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Frozen linear probe | 0.5847 | 0.7925 | 0.6726 | 0.6600 |
| LoRA, rank 8 | **0.7771** | **0.8111** | **0.7901** | **0.8408** |

| LoRA validation fold | Best epoch | Threshold | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Aperio CS2 | 4 | 0.25 | 0.6920 | **0.8693** | 0.7706 | 0.8347 |
| Hamamatsu S360 | 6 | 0.50 | **0.8431** | 0.7679 | **0.8037** | **0.8467** |
| Hamamatsu XR | 6 | 0.20 | 0.7961 | 0.7961 | 0.7961 | 0.8411 |

Relative to the frozen linear probe, LoRA improves macro F1 by 0.1175 and PR-AUC by 0.1808. The gain appears in all three folds, but this is still a one-seed, post-hoc validation pilot. Run seeds 1 and 2 before treating the difference as stable. Because the original locked tests have already been viewed, this method must be evaluated on a fresh external or newly reserved test set for an unbiased final claim. The runner saves each selected fold adapter and head alongside `validation_runs.csv` and `summary.json` for reproducibility.

### One-Seed LoRA Augmentation and GRL Ablation

The seed-0 LoRA pilot was extended with online HED stain augmentation, weak scanner-adversarial training, and their combination. HED jitter uses the teammate method (`sigma=0.05`, mode `both`) on every training batch before Phikon preprocessing; validation images remain unaugmented. GRL uses a balanced two-class scanner head with a 256-unit hidden layer and `lambda=0.01`. All other LoRA and optimization settings remain unchanged.

```powershell
# LoRA + online HED stain augmentation
.\.venv-det\Scripts\python.exe demo2\run_lora_probe_validation.py --model-key phikon_v2 --scale 40x --output-dir demo2\models\phikon_v2_40x_lora_stain_validation --seeds 0 --epochs 8 --patience 2 --batch-size 8 --eval-batch-size 16 --stain-augmentation --stain-sigma 0.05 --stain-mode both --offline

# LoRA + weak scanner-adversarial GRL
.\.venv-det\Scripts\python.exe demo2\run_lora_probe_validation.py --model-key phikon_v2 --scale 40x --output-dir demo2\models\phikon_v2_40x_lora_grl001_validation --seeds 0 --epochs 8 --patience 2 --batch-size 8 --eval-batch-size 16 --grl-strength 0.01 --offline

# LoRA + stain augmentation + GRL
.\.venv-det\Scripts\python.exe demo2\run_lora_probe_validation.py --model-key phikon_v2 --scale 40x --output-dir demo2\models\phikon_v2_40x_lora_stain_grl001_validation --seeds 0 --epochs 8 --patience 2 --batch-size 8 --eval-batch-size 16 --stain-augmentation --stain-sigma 0.05 --stain-mode both --grl-strength 0.01 --offline
```

| Seed-0 Phikon-v2 method | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| LoRA | 0.7771 | 0.8111 | 0.7901 | 0.8408 |
| LoRA + stain | **0.7789** | 0.8102 | 0.7925 | 0.8391 |
| LoRA + GRL (`lambda=0.01`) | 0.7711 | 0.8132 | 0.7910 | 0.8380 |
| LoRA + stain + GRL | 0.7783 | **0.8270** | **0.8017** | **0.8459** |

| Fold | LoRA F1 / AP | LoRA + stain F1 / AP | LoRA + GRL F1 / AP | Combined F1 / AP |
|---|---:|---:|---:|---:|
| Aperio CS2 | 0.7706 / 0.8347 | 0.7770 / 0.8369 | 0.7718 / 0.8309 | **0.7902 / 0.8464** |
| Hamamatsu S360 | **0.8037 / 0.8467** | 0.7850 / 0.8371 | 0.8036 / 0.8443 | 0.7991 / 0.8422 |
| Hamamatsu XR | 0.7961 / 0.8411 | 0.8155 / 0.8432 | 0.7977 / 0.8389 | **0.8156 / 0.8491** |

Stain augmentation alone and weak GRL alone change macro F1 by only +0.0024 and +0.0009, respectively, while each slightly reduces PR-AUC. Their combination is more promising: +0.0115 F1 and +0.0051 PR-AUC over LoRA alone, with a 0.0159 recall increase. The effect is scanner-dependent, and the combined Aperio fold selected the final allowed epoch rather than reaching an early-stopping plateau. These are single-seed exploratory results; seeds 1 and 2 and a longer matched epoch budget are required before calling the combined improvement stable. No locked test was evaluated.

### Three-Seed Virchow2 Confirmation

The selected Virchow2 comparison was repeated with seeds 1 and 2 using the fixed seed-0 settings. This is a paired comparison between LoRA alone and LoRA + stain + GRL; no hyperparameters were changed and no locked test was evaluated.

| Seed | LoRA F1 | Combined F1 | Delta F1 | LoRA PR-AUC | Combined PR-AUC | Delta PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8127 | 0.8309 | +0.0182 | 0.8745 | 0.8872 | +0.0127 |
| 1 | 0.8134 | 0.8225 | +0.0091 | 0.8758 | 0.8841 | +0.0083 |
| 2 | **0.8190** | 0.8154 | -0.0037 | 0.8792 | **0.8814** | +0.0022 |
| **Mean (seed SD)** | **0.8151 (0.0028)** | **0.8229 (0.0064)** | **+0.0079** | **0.8765 (0.0020)** | **0.8842 (0.0024)** | **+0.0077** |

The combined method improves PR-AUC in all three seeds and F1 in two of three. Across the nine seed-fold pairs, it improves F1 in seven and PR-AUC in eight. Mean paired improvements are positive on Aperio, S360, and XR, but combined F1 has greater seed variability than LoRA alone. The robust conclusion is improved ranking quality; the smaller F1 gain is promising but not fully seed-stable. A genuinely fresh external or newly reserved test set is still required for an unbiased generalization claim.

### Run the LoRA Ablation on All Encoders

`run_all_lora_models.py` runs the same four seed-0 methods for Hibou-B and Virchow2. Completed Phikon-v2 runs are included in the aggregate report and are not repeated. Hibou-B uses LoRA on separate `query` and `value` projections. Virchow2 uses its fused `qkv` projections with physical batch size 2 and four-step gradient accumulation, preserving effective batch size 8 on a 24 GB GPU.

From the repository root, run:

```powershell
.\.venv-det\Scripts\python.exe -u demo2\run_all_lora_models.py
```

The command runs sequentially and can be left open in its terminal. Each experiment gets a timestamped log in `demo2/logs/lora_all_models/`. After every successful run, aggregate metrics are rewritten to:

- `demo2/models/lora_all_models_summary.md`
- `demo2/models/lora_all_models_summary.csv`

The runner is resumable at experiment boundaries: rerunning the same command skips a method when its `summary.json` and all three fold adapters already exist. It stops immediately on an error and prints the relevant log path. To monitor the newest log from another PowerShell terminal:

```powershell
Get-Content (Get-ChildItem demo2\logs\lora_all_models\*.log | Sort-Object LastWriteTime | Select-Object -Last 1) -Wait
```

To rerun selected work, use `--models`, `--methods`, or `--force`; run `python demo2\run_all_lora_models.py --help` for the available values. These runs use source-scanner validation only and do not evaluate the previously viewed locked tests.

## Post-Hoc Held-Out LoRA Evaluation

The saved LoRA adapters can now be evaluated on the scanner excluded from their training fold. Each adapter keeps the checkpoint and threshold selected on the two source scanners; the evaluator does not retrain or retune them. It writes candidate predictions, per-scanner metrics, and image-level bootstrap intervals for each seed.

Run LoRA-only and combined Virchow2 separately:

```powershell
.\.venv-det\Scripts\python.exe demo2\evaluate_lora_heldout.py --checkpoint-dirs demo2\models\virchow2_40x_lora_validation demo2\models\virchow2_40x_lora_seeds12 --output-dir demo2\models\virchow2_40x_lora_posthoc_heldout --device cuda --offline

.\.venv-det\Scripts\python.exe demo2\evaluate_lora_heldout.py --checkpoint-dirs demo2\models\virchow2_40x_lora_stain_grl001_validation demo2\models\virchow2_40x_lora_stain_grl001_seeds12 --output-dir demo2\models\virchow2_40x_combined_posthoc_heldout --device cuda --offline
```

These results answer whether the already-selected LoRA models transfer to the excluded scanner. They must be labeled **post-hoc held-out-scanner results**, not a new locked test, because the earlier frozen-model test results were visible during later development. Do not use these outputs to change the model, threshold, or augmentation settings. A fresh external dataset remains necessary for an unbiased confirmatory claim.

## Independent Scanner-Invariance Diagnostic

The GRL training head is not itself evidence of scanner invariance. `evaluate_scanner_probe.py` freezes each trained encoder, extracts plain source-train and source-validation embeddings, and fits a new logistic-regression scanner classifier. It also repeats the probe separately within mitotic and non-mitotic candidates so class prevalence cannot create an easy scanner shortcut.

```powershell
.\.venv-det\Scripts\python.exe demo2\evaluate_scanner_probe.py --frozen-features demo2\cache\virchow2_40x.npz --checkpoint-dirs demo2\models\virchow2_40x_lora_validation demo2\models\virchow2_40x_lora_seeds12 demo2\models\virchow2_40x_lora_stain_grl001_validation demo2\models\virchow2_40x_lora_stain_grl001_seeds12 --output-dir demo2\models\virchow2_scanner_probe --device cuda --offline
```

Compare balanced accuracy against `0.50`, the chance level for the two source scanners. A combined model closer to `0.50` than LoRA alone contains less linearly recoverable source-scanner information. This supports only the statement that the representation is less predictive of the **two source scanners**; it does not prove invariance to the third scanner.

## Multi-Scale Virchow2 Probe

This experiment tests whether lower-magnification tissue context complements the local 40x candidate appearance. Virchow2 embeddings are extracted independently from aligned 20x and 40x patches. The fusion cache concatenates the two 2,560-dimensional vectors only after verifying exact agreement in candidate IDs, labels, image IDs, scanners, model ID, and pooling method.

```powershell
.\.venv-det\Scripts\python.exe demo2\cache_foundation_embeddings.py --model-key virchow2 --scale 20x --output demo2\cache\virchow2_20x.npz --device cuda

.\.venv-det\Scripts\python.exe demo2\combine_embedding_caches.py --inputs demo2\cache\virchow2_20x.npz demo2\cache\virchow2_40x.npz --output demo2\cache\virchow2_20x_40x.npz

.\.venv-det\Scripts\python.exe demo2\run_linear_probe_validation.py --features demo2\cache\virchow2_20x.npz --output-dir demo2\models\virchow2_20x_validation

.\.venv-det\Scripts\python.exe demo2\run_linear_probe_validation.py --features demo2\cache\virchow2_20x_40x.npz --output-dir demo2\models\virchow2_20x_40x_validation
```

| Virchow2 input | Feature dimension | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| 40x | 2,560 | **0.6943** | 0.7916 | **0.7384** | **0.7769** |
| 20x | 2,560 | 0.5503 | 0.7440 | 0.6315 | 0.6335 |
| 20x + 40x concatenation | 5,120 | 0.6561 | **0.7966** | 0.7194 | 0.7335 |

The 20x representation is substantially weaker for candidate classification. Concatenation recovers some performance and slightly raises recall over 40x alone, but loses 0.0382 precision, 0.0190 F1, and 0.0433 PR-AUC. The lower-magnification features therefore add noise under direct concatenation rather than useful complementary signal. The 40x linear probe remains the selected model, and the multi-scale variants are not promoted to locked testing.
