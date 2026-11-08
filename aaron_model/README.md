# Aaron's model: ResNet-18 add-ons (stain augmentation, GRL, soft labels, TTA, AdaBN)

This folder holds candidate classification (mitotic figure vs imposter) on the MIDOG++ breast
subset.

- **Evaluation:** leave-one-scanner-out. The held-out scanner is Aperio CS2, Hamamatsu S360
  or Hamamatsu XR.
- **Model:** ResNet-18 on 128 px patches (32 µm field).
- **Team metric:** F1 on all 50 cases of the held-out scanner. The threshold is chosen on
  source-scanner validation only.

The add-on study follows a pre-registered protocol, [`ADDONS_PROTOCOL.md`](ADDONS_PROTOCOL.md).
Phase 1 tried the add-ons with 3 seeds and picked "ResNet all" by source-validation F1.
Phase 2 is a 5-seed ablation of "all".

## Team grid (unseen scanner, mean of 3 folds × 5 seeds)

| Model | F1 as trained | F1 + TTA | F1 + AdaBN + TTA | AUROC as trained | AUROC + AdaBN + TTA |
|---|---|---|---|---|---|
| Baseline | 0.673 | 0.690 | 0.743 | 0.846 | 0.882 |
| + Stain (HED jitter, σ 0.05) | 0.713 | 0.714 | 0.759 | 0.867 | 0.894 |
| + GRL (scanner-adversarial head) | 0.687 | 0.690 | 0.748 | 0.854 | 0.886 |
| + Stain + GRL | 0.714 | 0.727 | **0.766** | 0.881 | **0.901** |
| **ResNet all** | **0.733** | **0.745** | **0.766** | **0.884** | 0.898 |

**ResNet all** combines H&E-only stain jitter, the GRL head and soft labels (rater-vote
fraction) during training. At test time it adds TTA (8 flips/rotations) and cross-fitted AdaBN.
With AdaBN + TTA it reaches F1 0.766 [0.740, 0.786], precision 0.729, recall 0.820 and
AUROC 0.898. A 5-seed ensemble reaches F1 0.785 and AUROC 0.908.

**Inference modes.** "As trained" and "+ TTA" use no target data. AdaBN re-estimates
BatchNorm statistics on *unlabelled* images of the held-out scanner, cross-fitted by case
(adapt on half the cases, predict the other half). It is therefore test-time adaptation, not
pure domain generalisation. Please compare models in the same mode.

![team grid](figures/ablation5_grid.png)

## Pre-registered confirmatory tests (F1, Holm over 6, two-level bootstrap B = 5000)

Each row removes one component from "all" at AdaBN + TTA; the last row is the headline.

| Component | ΔF1 [95 % CI] | p (Holm) |
|---|---|---|
| Stain augmentation | **+0.024 [0.011, 0.038]** | < 0.003 |
| GRL head | +0.012 [0.001, 0.025] | 0.097 |
| Soft labels | +0.007 [−0.006, 0.020] | 0.284 |
| TTA | **+0.015 [0.006, 0.024]** | 0.013 |
| Cross-fitted AdaBN (TTA kept) | +0.020 [0.001, 0.040] | 0.097 |
| **all (AdaBN + TTA) vs baseline as trained** | **+0.093 [0.064, 0.127]** | < 0.003 |

![confirmatory family](figures/ablation5_confirmatory.png)

Main exploratory findings (unadjusted CIs):

- **Training add-ons help much more as trained than after AdaBN + TTA.** Examples: soft labels
  +0.023 → +0.007; stain augmentation +0.044 → +0.024. In phase 1, EMA and Macenko normalisation
  help as trained and give no gain, or a loss, after AdaBN + TTA (Macenko −0.015).
- **"all" vs + Stain + GRL.** With AdaBN + TTA, the two differ by no more than about 0.015 F1
  (Δ −0.000 [−0.015, 0.014]). As trained, "all" is better by +0.020 [0.001, 0.038], mostly on
  S360.
- **Alternatives to AdaBN.** TENT and TENT+IM are within noise of AdaBN when TTA is used.
  AdaBN estimated on random tissue crops instead of candidate patches breaks the threshold:
  F1 −0.178, precision 0.42 at recall 0.998. UDA-DANN, which aligns to the target scanner during
  training, is no better than AdaBN.

Plain-language summary of these results: [`RESULTS.md`](RESULTS.md) ([PDF](RESULTS.pdf)).
The full methods write-up (`Aaron_Method_Report`, 8 November 2026) is not in this repo.

## Layout

```
ADDONS_PROTOCOL.md  pre-registered protocol of the add-on study (+ amendment)
RESULTS.md / .pdf   plain-language summary of the results
scripts/            code (paths are relative to this folder)
jobs/               job lists and the phase-1 / phase-2 / figure pipelines
results/            summary tables (CSV + markdown)
figures/            result plots
```

These are not in the repo, because they are too large or covered by the dataset licence:
`data/` (patches, MIDOG++ images), `checkpoints/`, per-run predictions (`results/preds*`)
and logs. Recreate them with the steps below.

## How to run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. crop the 4,435 candidates from the MIDOG++ breast TIFFs -> data/patches_128.npz, data/patches_meta.csv
.venv/bin/python scripts/prepare_patches.py --data-dir /path/to/midogpp_breast
#    optional inputs for the add-on study
.venv/bin/python scripts/prepare_macenko.py --data-dir /path/to/midogpp_breast       # -> data/patches_128_macenko.npz
.venv/bin/python scripts/prepare_random_crops.py --data-dir /path/to/midogpp_breast  # -> data/random_crops_128.npz

# 2. train: one held-out scanner, one configuration, one seed
.venv/bin/python scripts/train_patch.py --fold CS2 --variant stain_adv --stain-mode he_only --soft-labels --seed 0 --tag all
#    variants: baseline, stain, adv, stain_adv, mix, stain_mix
#    add-on flags: --sigma, --stain-mode {both,he_only,scale_only}, --ema, --soft-labels, --patches, --uda-half
#    a job list (TAG FOLD SEED [flags]) runs serially:
.venv/bin/python scripts/run_queue.py jobs/phase2.txt --workers 1

# 3. test-time steps on the trained checkpoints
.venv/bin/python scripts/eval_tta_adabn.py --tags all --seeds 0 1 2 3 4        # as trained / TTA / AdaBN
.venv/bin/python scripts/eval_adabn_crossfit.py --tags all --seeds 0 1 2 3 4   # cross-fitted AdaBN
.venv/bin/python scripts/eval_tent.py --tags all --seeds 0 1 2 3 4             # TENT / TENT+IM
.venv/bin/python scripts/eval_adabn_random.py --tags all --seeds 0 1 2 3 4     # AdaBN from random crops
.venv/bin/python scripts/eval_uda.py --base stain_uda --seeds 0 1 2            # merge the two UDA half-models
.venv/bin/python scripts/check_complete.py --tags all --seeds 0 1 2 3 4        # all outputs present and complete?

# 4. statistics and figures
.venv/bin/python scripts/summarize_addons.py --prefix ablation5 --seeds 0 1 2 3 4 --B 5000 \
    --tags all all_no_stain baseline --compare all@adabncf_tta:all_no_stain@adabncf_tta all@adabncf_tta:baseline@plain
.venv/bin/python scripts/holm_confirmatory.py --pairwise results/ablation5_pairwise.csv --B 5000
.venv/bin/python scripts/summarize_ensemble.py --variants all --modes plain adabncf_tta --seeds 0 1 2 3 4 --prefix ensemble5
zsh jobs/make_figures.sh
```

`jobs/run_phase1.sh` and `jobs/run_phase2.sh` are the exact pipelines that produced the results.
Step 1 of `run_phase2.sh` waits for a one-off integrity re-check that is not part of this repo.
Skip that step when re-running.

**Run GPU jobs one at a time.** On Apple MPS, concurrent GPU processes silently corrupted
outputs, so all GPU jobs here run serially (`--workers 1`). `scripts/compare_preds.py` checks a
re-run against stored predictions.

| Script | What it does |
|---|---|
| `prepare_patches.py` | crops each annotation at a fixed physical size (32 µm) on every scanner |
| `prepare_macenko.py`, `prepare_random_crops.py` | Macenko-normalised patches; random tissue crops for random-crop AdaBN |
| `train_patch.py` | ResNet-18 training with stain jitter, GRL head, MixStyle, soft labels, EMA, UDA; threshold picked on source validation by F1 |
| `addon_common.py` | shared folds, defaults, model loading and target halves for cross-fitting |
| `eval_tta_adabn.py`, `eval_adabn_crossfit.py` | predictions as trained / + TTA / + AdaBN (+ cross-fitted) |
| `eval_tent.py`, `eval_adabn_random.py`, `eval_uda.py` | test-time alternatives: TENT / TENT+IM, random-crop AdaBN, UDA-DANN |
| `summarize_addons.py` | two-level bootstrap (slides within fold, seeds shared by both arms), paired comparisons, source-val selection table |
| `holm_confirmatory.py` | the six pre-registered tests with Holm adjustment |
| `select_all.py` | applies the selection rule and writes the phase-2 job list |
| `summarize_ensemble.py` | seed ensembles vs the average single model |
| `run_queue.py`, `check_complete.py`, `compare_preds.py` | serial job queue, completeness gate, re-run comparison |
| `plot_addons.py` | heatmap and dot-plot figures (`jobs/make_figures.sh`) |
| `summarize_*.py`, `plot_tta.py` | earlier (5 Nov) statistics and figures |
| `phikon_probe.py`, `prepare_patches_224.py` | frozen Phikon foundation-model baseline (negative result) |

## Results files

| File | Content |
|---|---|
| `results/ablation5_*` | phase 2, 5 seeds: team grid, "all", all-minus-one, supplementary arm; `_confirmatory` = Holm table |
| `results/ablation5_extra_*` | more phase-2 contrasts (e.g. each model at AdaBN + TTA vs the baseline at AdaBN + TTA) |
| `results/ablation5_alt_*` | TENT / TENT+IM / random-crop AdaBN on "all" |
| `results/ablation3_*` | 3-seed comparisons including UDA-DANN |
| `results/addons_train_*`, `results/addons_grid_*` | phase 1 (3 seeds): training add-ons; grid with TENT and random-crop AdaBN |
| `results/addons_selection.json` | source-validation F1 of each candidate and the selected "all" configuration |
| `results/ensemble5_*` | 5-seed ensembles |
| `results/*_runs.csv` | per run (fold × tag × seed × mode) metrics |

## Earlier results (5 November, 3 seeds)

The first version of this folder reported +Stain with AdaBN + TTA at F1 0.765 (3 seeds), and a
3-seed ensemble at F1 0.781. Its other findings:

- MixStyle gave no gain: F1 +0.014, with a CI that crosses 0.
- Frozen Phikon features with a logistic probe stayed far below the CNN (AUROC 0.68–0.77).

Files: `results/tta_*`, `results/ensemble_*`, `results/mixstyle_*`, `results/summary_*` and
`figures/tta_adabn_results.png`.

With seeds 3–4 added, the as-trained and TTA numbers moved by up to 0.016 F1. The AdaBN numbers
moved by at most 0.007. The table above supersedes the older numbers.
