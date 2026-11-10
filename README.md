# General Mitosis Detection Across Scanners

This repository contains a collaborative study of scanner-robust mitosis analysis in
histopathology images. The project uses the MIDOG++ breast-cancer data to investigate how
stain variation and scanner domain shift affect mitotic-figure models.

The work covers two related tasks:

- **Candidate classification:** classify an annotated candidate patch as mitotic or
  non-mitotic.
- **Object detection:** locate mitotic figures in larger image regions using bounding boxes.

Results from these tasks are reported separately because candidate classification assumes
that candidate locations are already known, while object detection must also find them.

## Problem

Mitotic figures are important indicators of tumor activity, but they are small, visually
variable, and easily confused with non-mitotic cells. Models trained with images from one
scanner can also learn scanner-specific color and acquisition patterns. Their performance
may therefore decline when slides come from a scanner that was not represented during
training.

The central question is:

> Which training and adaptation methods improve mitosis recognition on an unseen scanner,
> and do improvements on cropped candidate patches transfer to object detection?

The experiments focus on three breast-cancer scanner domains: Aperio CS2, Hamamatsu S360,
and Hamamatsu XR. Leave-one-scanner-out (LOSO) evaluation trains on two domains and evaluates
on the third, rotating the held-out scanner across three folds.

## Approach

The team studies the problem at increasing levels of complexity:

1. **Patch and candidate baselines:** measure scanner shift using cropped mitotic and
   non-mitotic candidates.
2. **Stain robustness:** perturb H&E color channels during training or normalize image color
   to reduce reliance on scanner-specific appearance.
3. **Scanner-adversarial learning:** attach a scanner-prediction head through a
   gradient-reversal layer (GRL) to discourage scanner information in learned features.
4. **Test-time adaptation:** use test-time augmentation (TTA) and cross-fitted adaptive batch
   normalization (AdaBN). AdaBN uses unlabeled target-scanner candidates, so it is reported
   as adaptation rather than pure domain generalization.
5. **Pathology foundation models:** compare frozen Phikon-v2, Hibou-B, and Virchow2 features,
   then adapt their encoders with rank-8 LoRA.
6. **Object detection:** train Faster R-CNN on image tiles to test whether patch-level gains
   survive candidate localization and whole-region false positives.

Thresholds and checkpoints are selected on source-scanner validation data in the locked
protocols. Multiple seeds and image-level bootstrap intervals are used where available.

## Main results

### Candidate classification

| Study | Method | Main result | Interpretation |
|---|---|---:|---|
| Aaron, locked 3-fold x 5-seed study | ResNet-18 baseline, as trained | F1 0.673 | Reference candidate classifier |
| Aaron, locked 3-fold x 5-seed study | Stain + GRL + soft labels, as trained | F1 0.733 | Training add-ons improved F1 by 0.060 |
| Aaron, locked 3-fold x 5-seed study | Full method with AdaBN + TTA | F1 0.766, 95% CI 0.740-0.786 | Stain augmentation and TTA had statistically supported contributions |
| Jun, post-hoc held-out evaluation | Virchow2 LoRA | F1 0.8062, PR-AUC 0.8765 | Strong transfer from source validation to held-out scanners |
| Jun, post-hoc held-out evaluation | Virchow2 LoRA + stain + GRL | F1 0.8073, PR-AUC 0.8792 | Only +0.0010 F1 over LoRA; no meaningful held-out gain |

Aaron's full method reached precision 0.729, recall 0.820, and AUROC 0.898 with AdaBN +
TTA. The confirmatory ablation attributed `+0.024` F1 to stain augmentation and `+0.015`
F1 to TTA after multiplicity correction. GRL and soft labels had smaller, statistically
uncertain contributions under that inference setting.

Jun's LoRA results are post-hoc because earlier held-out results were visible during model
development. They should not be treated as an untouched confirmatory test or directly used
to claim improvement over the earlier frozen Virchow2 held-out F1 of 0.707. An independent
scanner probe achieved approximately 1.0 balanced accuracy on frozen, LoRA, and combined
representations, showing that weak GRL did **not** remove linearly decodable scanner
information.

### Object detection

The exploratory Faster R-CNN study reported mean LOSO F1 of 0.728 for the baseline and 0.698
with strong color jitter. On Hamamatsu S360, Reinhard-style RGB normalization produced F1
0.715 versus 0.714 for the baseline, with higher precision but lower recall.

These detection values are **development results, not unbiased final generalization
estimates**. A protocol audit found that the original run selected checkpoints and thresholds
on the held-out scanner and used annotation-centered test tiles. Corrected code now performs
source-only model selection, deterministic exhaustive test tiling, one-to-one matching, and
seeded training, but the locked detection experiment still needs to be rerun.

## Overall findings

- Scanner shift is measurable and remains strongly encoded even in adapted foundation-model
  representations.
- Stain augmentation helped simpler patch classifiers, but augmentation alone did not
  reliably improve the exploratory object detector.
- TTA and target-aware AdaBN produced substantial candidate-classification gains, although
  AdaBN changes the setting from pure domain generalization to test-time adaptation.
- Virchow2 with LoRA is Jun's strongest current candidate-classification approach.
- Adding stain augmentation and weak GRL to Virchow2 LoRA did not materially improve
  held-out performance or scanner invariance.
- Candidate classification is only one stage of a clinical detection workflow. The next
  evidence needed is a locked, exhaustively tiled detection study followed by evaluation on
  genuinely external pathology data.

## Team contributions

### Aaron

[`aaron_model/`](aaron_model/) contains ResNet-18 candidate-classification experiments,
including stain augmentation, scanner-adversarial learning with a gradient-reversal layer
(GRL), soft labels, test-time augmentation (TTA), and adaptive batch normalization (AdaBN).

- [Methods and usage](aaron_model/README.md)
- [Results](aaron_model/RESULTS.md)
- [Experiment protocol](aaron_model/ADDONS_PROTOCOL.md)

### Jun

[`jun_model/`](jun_model/) contains classical baselines, Faster R-CNN detection experiments,
foundation-model linear probing and LoRA, stain augmentation, GRL experiments, and reusable
scanner-generalization workflows.

- [Contribution overview](jun_model/README.md)
- [Demo 1: patch baselines and object detection](jun_model/demo1/README.md)
- [Demo 2: foundation-model candidate classification](jun_model/demo2/README.md)
- [Demo 2 detailed report](jun_model/demo2/reports/demo2_detailed_summary.md)

## Repository layout

```text
aaron_model/       Aaron's experiments, reports, and reproducibility scripts
jun_model/         Jun's experiments, reports, prepared metadata, and workflows
midogpp/           Upstream MIDOG++ dataset infrastructure and original model code
  configs/         Original training configuration
  databases/       COCO JSON and SlideRunner SQLite annotations
  images/          Local downloaded images (not committed)
  slide/           Whole-slide image loading utilities
  utils/           Detection, metric, callback, and NMS utilities
  Setup.ipynb      Image-download notebook
  main.py          Original training/inference entry point
```

Generated images, patches, embeddings, checkpoints, logs, and model weights are excluded
from Git. Each contribution directory documents how to recreate the artifacts it uses.

## How to run each directory

Commands below assume Windows PowerShell. Run them from the repository root unless a step
explicitly changes directories.

### `midogpp/`: original MIDOG++ data and code

This directory contains the upstream annotation database, image downloader, loaders, and
original training and inference modules used as shared infrastructure by the team.

The pinned upstream dependencies require a Python 3.8 environment:

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r midogpp\requirements.txt
```

Open [`midogpp/Setup.ipynb`](midogpp/Setup.ipynb) in VS Code, select `.venv` as its kernel,
and run all cells to download the approximately 65 GB image collection into
`midogpp/images/`. This legacy environment reproduces the upstream dependencies; use
`.venv-det` for the modern CUDA detection and foundation-model pipelines. The original
[`midogpp/main.py`](midogpp/main.py) is retained as upstream reference code rather than a
turnkey project command: its training call is commented out, and its inference path expects
configured Weights & Biases artifacts. Run that original pipeline from inside `midogpp/`
after configuring [`midogpp/configs/all.yaml`](midogpp/configs/all.yaml).

### `aaron_model/`: ResNet-18 candidate classification

Aaron's commands use paths relative to `aaron_model/`, so enter that directory first and
create its own environment:

```powershell
Set-Location aaron_model
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$DATA_DIR = "D:\path\to\midogpp_breast"
.\.venv\Scripts\python.exe scripts\prepare_patches.py --data-dir $DATA_DIR
.\.venv\Scripts\python.exe scripts\train_patch.py --fold CS2 --variant stain_adv --stain-mode he_only --soft-labels --seed 0 --tag all
.\.venv\Scripts\python.exe scripts\run_queue.py jobs\phase2.txt --workers 1
```

The exact phase pipelines, evaluation commands, and statistics commands are documented in
[`aaron_model/README.md`](aaron_model/README.md). Return to the repository root with
`Set-Location ..` before running Jun's commands.

### `jun_model/demo1/`: patch baselines and Faster R-CNN

Prepare the scanner split and patch data:

```powershell
.\.venv\Scripts\python.exe jun_model\demo1\demo1_make_scanner_split.py
.\.venv\Scripts\python.exe jun_model\demo1\demo1_build_patch_dataset.py
.\.venv\Scripts\python.exe jun_model\demo1\demo1_run_loso_pixel_experiments.py
```

Run the corrected Faster R-CNN LOSO pipeline with the CUDA detection environment:

```powershell
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_prepare_detection_coco.py
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_build_all_detection_tiles.py
.\.venv-det\Scripts\python.exe jun_model\demo1\demo1_run_all_detection_loso.py --device cuda --epochs 8 --batch-size 1 --lr 1e-4 --clip-grad-norm 1.0 --seed 42 --skip-build-tiles --output-suffix _locked_v2 --summary-dir jun_model\demo1\models\detection_loso_summary_locked_v2
```

This is a long GPU run. See [`jun_model/demo1/README.md`](jun_model/demo1/README.md) for
visual checks, smoke tests, individual training commands, and evaluation details.

### `jun_model/demo2/`: foundation models and LoRA

Prepare manifests and physical-scale candidate patches, then install the additional
foundation-model dependencies:

```powershell
.\.venv-det\Scripts\python.exe -m pip install -r jun_model\demo2\requirements-foundation.txt
.\.venv-det\Scripts\python.exe jun_model\demo2\prepare_demo2_data.py
.\.venv-det\Scripts\python.exe jun_model\demo2\preview_patch_scales.py
.\.venv-det\Scripts\python.exe jun_model\demo2\extract_candidate_patches.py
```

The gated Hibou-B and Virchow2 models require accepted Hugging Face access and authentication:

```powershell
.\.venv-det\Scripts\hf.exe auth login
.\.venv-det\Scripts\python.exe jun_model\demo2\cache_foundation_embeddings.py --model-key virchow2 --scale 40x --output jun_model\demo2\cache\virchow2_40x.npz --device cuda
.\.venv-det\Scripts\python.exe jun_model\demo2\run_linear_probe_validation.py --features jun_model\demo2\cache\virchow2_40x.npz --output-dir jun_model\demo2\models\virchow2_40x_validation
```

Run the reproducible cross-model LoRA experiment queue with:

```powershell
.\.venv-det\Scripts\python.exe -u jun_model\demo2\run_all_lora_models.py
```

That runner is resumable and writes timestamped logs under `jun_model/demo2/logs/`. See
[`jun_model/demo2/README.md`](jun_model/demo2/README.md) for model access requirements,
locked configurations, held-out evaluation, scanner probes, and all ablation commands.

### `jun_model/mitosis_scanner_workflow/`: reusable orchestration

This directory packages the normalization, augmentation, and detection stages for reuse by
other collaborators. Validate its configuration before a run:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py validate
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py --dry-run train --holdout Hamamatsu_S360 --variant reinhard_rgb
```

Its interface and artifact contract are documented in
[`jun_model/mitosis_scanner_workflow/README.md`](jun_model/mitosis_scanner_workflow/README.md).

## Dataset setup notes

The underlying dataset is MIDOG++, a multi-domain mitotic-figure dataset spanning tumor
types, laboratories, staining conditions, and slide scanners. This project primarily uses
its breast-cancer scanner domains.

1. Create and activate a compatible Python environment.
2. Install the original MIDOG++ dependencies:

   ```powershell
   python -m pip install -r midogpp\requirements.txt
   ```

3. Run [`midogpp/Setup.ipynb`](midogpp/Setup.ipynb) to download the original images into
   `midogpp/images/`.
4. Follow the README in `aaron_model/` or `jun_model/` for the selected experiment. Those
   experiments may use separate environments and additional dependencies.

The full image download is approximately 65 GB. Dataset images and derived pathology crops
must remain local and should not be committed to Git.

## Evaluation principles

- Scanner generalization uses leave-one-scanner-out (LOSO) evaluation.
- Model selection and threshold tuning must use only source-scanner training or validation
  data.
- The held-out scanner is used once for final evaluation.
- Candidate-classification metrics must not be presented as full-image detection metrics.
- Comparisons should use the same folds, seeds, adaptation setting, and evaluation protocol.

## MIDOG++ attribution

This repository builds on the original MIDOG++ code and dataset from
[DeepMicroscopy/MIDOGpp](https://github.com/DeepMicroscopy/MIDOGpp).

If you use MIDOG++ in research, cite:

> Aubreville, M., Wilm, F., Stathonikos, N. et al. A comprehensive multi-domain dataset for
> mitotic figure detection. *Scientific Data* 10, 484 (2023).
> https://doi.org/10.1038/s41597-023-02327-4
