# Agent Instructions: Scanner Robustness Workflow

## Objective

Help collaborators run and extend reproducible scanner-generalization experiments for mitotic figure detection. The primary comparison is baseline versus color augmentation versus stain/color normalization under leave-one-scanner-out evaluation.

## Start Here

1. Read `README.md` and `configs/default.json`.
2. Run `python jun_model/mitosis_scanner_workflow/run_workflow.py validate --require-tiles`.
3. Use `--dry-run` before starting a training matrix.
4. Use `configs/smoke.json` before any full experiment.

## Non-Negotiable Experimental Rules

- Never use the held-out scanner to fit normalization targets, tune preprocessing, or choose hyperparameters.
- Apply random color augmentation only to training images.
- Apply deterministic normalization consistently to training and evaluation images.
- Keep the same detector architecture, optimizer settings, tile data, IoU rule, and score grid when comparing preprocessing variants.
- Report precision, recall, F1, TP, FP, and FN. Do not report loss alone as detection performance.
- Select checkpoints and score thresholds only on training-scanner validation slides.
- Validation and test must use exhaustive grid tiles covering every source slide.
- Describe the task as exhaustive tile-grid detection. Slide-level prediction merging is not included yet.
- Do not commit TIFF data, generated tiles, model weights, virtual environments, or `artifacts/`.
- Do not modify or delete existing model artifacts unless the collaborator explicitly requests it.

## Supported Experiment Names

Holdouts:

```text
Aperio_CS2
Hamamatsu_S360
Hamammatsu_XR
```

The `Hamammatsu_XR` spelling matches the source metadata and must remain unchanged in paths and config keys.

Variants:

```text
baseline
mild_color_jitter
strong_color_jitter
reinhard_rgb
```

## Agent Workflow

For a requested experiment:

1. Validate the repository and tile annotations.
2. Print the planned command with `--dry-run`.
3. Run the smoke configuration.
4. Inspect training completion and `eval/metrics.json`.
5. Run the full configuration only after the smoke test succeeds.
6. Run `summarize` and report the artifact paths.
7. Compare each method against the matching scanner baseline, not against a different holdout.

## Adding A New Method

When adding Macenko, Vahadane, adversarial learning, or another method:

- give it a unique config variant name
- document whether it is deterministic preprocessing or random augmentation
- record all fitted target statistics and their provenance
- add train/evaluation parity checks
- preserve the existing variants for fair comparison
- update both `README.md` and this file

## Completion Criteria

An experiment is complete only when:

- training exited successfully
- the intended checkpoint exists
- evaluation used the same normalization mode as training
- locked test `metrics.json` and `metrics.csv` exist
- `protocol_summary.json` records the validation-selected checkpoint and threshold
- the result appears in `summary.csv` and `summary.md`
- limitations and any deviations from the default config are documented
