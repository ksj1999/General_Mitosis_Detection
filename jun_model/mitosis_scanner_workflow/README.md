# Mitosis Scanner Robustness Workflow

This directory packages the tested MIDOG++ workflow for collaborators and coding agents. It compares color augmentation and RGB normalization for mitotic figure detection under scanner domain shift.

The detector is Faster R-CNN with a ResNet50-FPN backbone. It operates on `512 x 512` tiles generated from the larger pathology images. This is full object detection on pathology tiles; whole-slide stitching and slide-level aggregation are not included yet.

## What Is Included

- one command-line entry point: `run_workflow.py`
- baseline, mild jitter, strong jitter, and Reinhard RGB variants
- leave-one-scanner-out experiment definitions
- smoke-test and full experiment configurations
- automatic precision, recall, F1, TP, FP, and FN summaries
- `AGENTS.md` with constraints and handoff instructions for coding agents

Data, generated tiles, model weights, and experiment outputs are intentionally excluded from Git.

## Expected Repository Layout

This directory reuses the tested implementation in `jun_model/demo1/`:

```text
MIDOGpp/
  jun_model/demo1/
    demo1_build_all_detection_tiles.py
    demo1_train_detection_frcnn.py
    demo1_evaluate_detection_frcnn.py
    detection_dataset.py
    detection_coco/
    detection_tiles/
  jun_model/mitosis_scanner_workflow/
```

## Environment

The verified local environment was:

```text
Python 3.10.11
PyTorch 2.5.1 + CUDA 12.1
torchvision 0.20.1 + CUDA 12.1
```

Install the correct PyTorch build for the collaborator's CPU or GPU first, then install the remaining requirements:

```powershell
python -m pip install -r jun_model\mitosis_scanner_workflow\requirements.txt
```

## Quick Start

Validate scripts and existing tile data:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py validate --require-tiles
```

Preview a command without running it:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py --dry-run train --holdout Hamamatsu_S360 --variant reinhard_rgb
```

Run a fast smoke test:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py --config jun_model\mitosis_scanner_workflow\configs\smoke.json run --holdout Aperio_CS2 --variant baseline
```

Run one full experiment:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py run --holdout Hamamatsu_S360 --variant reinhard_rgb
```

Run all four variants for one held-out scanner:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py matrix --holdouts Hamamatsu_S360 --variants all
```

Run the complete three-scanner matrix. This trains 12 detectors and may take substantial GPU time:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py matrix --holdouts all --variants all
```

Create a combined result table after evaluation:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py summarize
```

## Prepare Detection Tiles

If the prepared data does not exist, run:

```powershell
.\.venv-det\Scripts\python.exe jun_model\mitosis_scanner_workflow\run_workflow.py prepare
```

This command creates train/validation/test scanner splits and then builds tiles. Training uses sampled positive/negative tiles. Validation and test use an exhaustive non-overlapping grid covering every source slide, including slides without mitoses.

## Variants

| Variant | Training augmentation | Train normalization | Test normalization |
| --- | --- | --- | --- |
| `baseline` | none | none | none |
| `mild_color_jitter` | mild random jitter | none | none |
| `strong_color_jitter` | strong random jitter | none | none |
| `reinhard_rgb` | none | Reinhard-style RGB | Reinhard-style RGB |

Color jitter is applied only to training images. Normalization must be applied identically during training and evaluation.

Mild jitter:

```text
brightness 0.08, contrast 0.08, saturation 0.05, hue 0.01
```

Strong jitter:

```text
brightness 0.18, contrast 0.18, saturation 0.12, hue 0.03
```

The current Reinhard-style RGB target is defined in `jun_model/demo1/detection_dataset.py`:

```text
target mean = [185, 135, 175]
target std  = [45, 35, 40]
```

## Evaluation Design

Use leave-one-scanner-out evaluation:

| Training domains | Held-out domain |
| --- | --- |
| S360 + XR | Aperio CS2 |
| Aperio + XR | Hamamatsu S360 |
| Aperio + S360 | Hamammatsu XR |

Do not calculate normalization targets, thresholds, or other preprocessing statistics using the held-out scanner. Select score thresholds on validation data when making a final scientific comparison.

## Output Layout

Generated files are written below the ignored artifact directory:

```text
jun_model/mitosis_scanner_workflow/artifacts/
  <held-out-scanner>/
    <variant>/
      config.json
      best.pt
      final.pt
      training_log.csv
      training_log.json
      locked_protocol/
        protocol_summary.json
        protocol_summary.md
        validation/
        test/
          metrics.csv
          metrics.json
  summary.csv
  summary.md
```

The default configuration saves only `best.pt` and `final.pt` to avoid generating tens of gigabytes of duplicate checkpoints. Both are compared on validation F1, and the selected checkpoint/threshold is evaluated once on the held-out scanner. Set `save_every` to `1` only when epoch-by-epoch validation selection is needed.

## Extending The Workflow

Add a preprocessing method by:

1. Implementing it in `jun_model/demo1/detection_dataset.py`.
2. Exposing the option in the training and evaluation scripts.
3. Adding a named variant to `configs/default.json`.
4. Applying deterministic normalization to both train and test data.
5. Applying random augmentation only to training data.
6. Running the smoke configuration before a full LOSO matrix.

See `AGENTS.md` for the machine-agent contract and safety constraints.
