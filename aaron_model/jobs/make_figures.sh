#!/bin/zsh
# Figures of the add-on study (CPU). Needs the summaries written by jobs/run_phase2.sh, the B = 5000 re-run of
# ablation5 and scripts/holm_confirmatory.py (see ADDONS_PROTOCOL.md).
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# phase 1: each training add-on vs its reference, as trained and with AdaBN+TTA (3 seeds)
for M in f1 auroc; do
  $PY scripts/plot_addons.py ablation --pairwise results/addons_train_pairwise.csv --modes plain adabncf_tta --metrics $M \
    --comparisons "stain_s002 vs stain:σ 0.02 (vs 0.05)" "stain_s010 vs stain:σ 0.10 (vs 0.05)" \
      "stain_he vs stain:H&E only (DAB fixed)" "stain_scale02 vs stain:Scale only, σ 0.2" "stain_ema vs stain:EMA of weights" \
      "stain_soft vs stain:Soft labels (rater votes)" "macenko_stain vs stain:Macenko + stain aug" \
      "macenko vs baseline:Macenko (vs baseline)" \
    --out figures/addons_train_$M.png \
    --note "Each add-on minus its reference (+Stain with σ 0.05, or baseline), same seeds; 3 held-out scanners x 3 seeds; two-level bootstrap 95 % CI. Filled dot: CI excludes 0 (unadjusted, exploratory)."
done

# phase 2: team grid, 5 seeds
$PY scripts/plot_addons.py grid --summary results/ablation5_summary.csv \
  --rows baseline:Baseline stain:"+Stain" adv:"+GRL" stain_adv:"+Stain+GRL" all:"ResNet all" \
  --cols plain:"as trained" tta:"+TTA" adabncf:"+AdaBN" adabncf_tta:"+AdaBN+TTA" --out figures/ablation5_grid.png \
  --note "Mean of 3 held-out scanners x 5 seeds. TTA: 8 flips/rotations. AdaBN: BatchNorm statistics re-estimated on unlabelled target images, cross-fitted by case. Threshold from source val. ResNet all = H&E-only stain jitter + GRL + soft labels. Bold: best in panel (ties at 3 decimals)."

# phase 2: pre-registered confirmatory family (F1 dots filled by Holm-adjusted p)
$PY scripts/plot_addons.py ablation --pairwise results/ablation5_pairwise.csv --holm results/ablation5_confirmatory.csv \
  --comparisons "all@adabncf_tta - all_no_stain@adabncf_tta:Stain augmentation" \
    "all@adabncf_tta - all_no_grl@adabncf_tta:GRL scanner head" \
    "all@adabncf_tta - all_no_soft@adabncf_tta:Soft labels" \
    "all@adabncf_tta - all@adabncf:TTA" \
    "all@adabncf_tta - all@tta:Cross-fitted AdaBN" \
    "all@adabncf_tta - baseline@plain:ResNet all vs baseline" \
  --out figures/ablation5_confirmatory.png \
  --note "ResNet all (AdaBN+TTA) minus the same pipeline without one component; the last row compares it with the baseline as trained. 3 held-out scanners x 5 seeds, paired two-level bootstrap B = 5000. F1: filled = Holm-adjusted p < 0.05 (6 pre-registered tests). AUROC (exploratory): filled = 95 % CI excludes 0."

# phase 2: training components at both inference modes (exploratory)
$PY scripts/plot_addons.py ablation --pairwise results/ablation5_pairwise.csv --modes plain adabncf_tta --metrics f1 \
  --comparisons "all vs all_no_stain:Stain augmentation" "all vs all_no_grl:GRL scanner head" "all vs all_no_soft:Soft labels" \
    "all vs all_stain_default:H&E-only vs default stain jitter" "all vs stain_adv:Soft labels + H&E-only (vs +Stain+GRL)" \
  --out figures/ablation5_components.png \
  --note "ResNet all minus the variant without the component, both at the same inference mode. 3 held-out scanners x 5 seeds, paired two-level bootstrap B = 5000. Filled: 95 % CI excludes 0 (unadjusted, exploratory)."
