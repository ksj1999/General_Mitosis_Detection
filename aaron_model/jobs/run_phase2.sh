#!/bin/zsh
# Phase 2 of the add-on study (ADDONS_PROTOCOL.md "Final ablation" + amendment of 2026-11-08), strictly serial on the GPU.
# Job list from scripts/select_all.py (results/addons_selection.json): "all", all minus one training component,
# all with the default stain config (supplementary), and the team grid, seeds 0-4 x 3 folds.
# Launch:  caffeinate -i zsh jobs/run_phase2.sh > logs/phase2_chain.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TAGS=(all all_no_stain all_no_grl all_no_soft all_stain_default baseline stain adv stain_adv)
SEEDS=(0 1 2 3 4)
ts() { date +%H:%M:%S; }
die() { echo "[$(ts)] PHASE2_FAILED: $*"; exit 1; }

# ---- 1. wait for the integrity re-verification (GPU stays serial) ----
until grep -q VERIFY_DONE logs/verify_concurrent.log 2>/dev/null; do
  pgrep -f jobs/verify_concurrent.sh >/dev/null || grep -q VERIFY_DONE logs/verify_concurrent.log || die "verify died without VERIFY_DONE"
  sleep 30
done
grep -q Traceback logs/verify_concurrent.log && die "Traceback in logs/verify_concurrent.log"
echo "[$(ts)] verify summary:"; grep -E "compared|replaced|CORRUPT|ROW MISMATCH|no stored" logs/verify_concurrent.log
# thresholds of the adv TTA/AdaBN sidecars come from source-val passes run in the concurrent window: compare them too
$PY scripts/check_adv_thresholds.py || die "adv threshold check"
# phase-1 grid tables may be stale if verify replaced inputs: rebuild them (CPU)
$PY scripts/summarize_addons.py --tags baseline stain adv stain_adv stain_mix \
  --modes plain tta adabncf adabncf_tta tent tent_tta tentim tentim_tta adabnrand adabnrand_tta \
  --compare stain@plain:baseline@plain adv@plain:baseline@plain stain_adv@plain:baseline@plain stain_adv@plain:stain@plain \
    baseline@adabncf_tta:baseline@plain stain@adabncf_tta:stain@plain adv@adabncf_tta:adv@plain stain_adv@adabncf_tta:stain_adv@plain \
    stain@tent_tta:stain@adabncf_tta stain@tentim_tta:stain@adabncf_tta stain@adabnrand_tta:stain@adabncf_tta \
    stain@tent:stain@plain stain@adabnrand:stain@plain stain_adv@adabncf_tta:stain@adabncf_tta \
  --prefix addons_grid --overwrite > logs/addons_grid_rebuild.log 2>&1 || die "grid rebuild"

# ---- 2. training (one automatic retry: the queue skips finished jobs) ----
echo "[$(ts)] training queue"
caffeinate -i $PY scripts/run_queue.py jobs/phase2.txt --workers 1 || \
  { echo "[$(ts)] queue had failures, retrying unfinished jobs"; caffeinate -i $PY scripts/run_queue.py jobs/phase2.txt --workers 1; } || \
  die "training queue"

# ---- 3. evaluation ----
echo "[$(ts)] TTA / AdaBN"
caffeinate -i $PY scripts/eval_tta_adabn.py --tags $TAGS --seeds $SEEDS || die "eval_tta_adabn"
echo "[$(ts)] cross-fitted AdaBN"
caffeinate -i $PY scripts/eval_adabn_crossfit.py --tags $TAGS --seeds $SEEDS || die "eval_adabn_crossfit"
echo "[$(ts)] TENT / random-crop AdaBN on all (alternatives)"
caffeinate -i $PY scripts/eval_tent.py --tags all --seeds $SEEDS || die "eval_tent"
caffeinate -i $PY scripts/eval_adabn_random.py --tags all --seeds $SEEDS || die "eval_adabn_random"

# ---- 4. completeness gate (CPU) ----
$PY scripts/check_complete.py --tags $TAGS --seeds $SEEDS --extra-tags all --extra tent rand || die "incomplete outputs"

# ---- 5. summaries (CPU; same rng seed -> shared bootstrap draws across invocations) ----
echo "[$(ts)] summaries"
$PY scripts/summarize_addons.py --prefix ablation5 --seeds $SEEDS --overwrite \
  --tags all all_no_stain all_no_grl all_no_soft all_stain_default baseline stain adv stain_adv \
  --modes plain tta adabncf adabncf_tta \
  --compare all@adabncf_tta:all_no_stain@adabncf_tta all@adabncf_tta:all_no_grl@adabncf_tta \
    all@adabncf_tta:all_no_soft@adabncf_tta all@adabncf_tta:all@adabncf all@adabncf_tta:all@tta \
    all@adabncf_tta:baseline@plain \
    all@adabncf_tta:all_stain_default@adabncf_tta \
    all@adabncf_tta:all@plain all@plain:all_no_stain@plain all@plain:all_no_grl@plain all@plain:all_no_soft@plain \
    all@plain:all_stain_default@plain \
    all@adabncf_tta:stain@plain all@adabncf_tta:adv@plain all@adabncf_tta:stain_adv@plain \
    all@adabncf_tta:baseline@adabncf_tta all@adabncf_tta:stain@adabncf_tta all@adabncf_tta:adv@adabncf_tta \
    all@adabncf_tta:stain_adv@adabncf_tta all@plain:stain_adv@plain \
    stain@plain:baseline@plain adv@plain:baseline@plain stain_adv@plain:baseline@plain stain_adv@plain:stain@plain \
    baseline@adabncf_tta:baseline@plain stain@adabncf_tta:stain@plain adv@adabncf_tta:adv@plain \
    stain_adv@adabncf_tta:stain_adv@plain \
  > logs/ablation5.log 2>&1 || die "ablation5 summary"
$PY scripts/summarize_addons.py --prefix ablation5_alt --seeds $SEEDS --overwrite --tags all \
  --modes plain adabncf adabncf_tta tent tent_tta tentim tentim_tta adabnrand adabnrand_tta \
  --compare all@tent_tta:all@adabncf_tta all@tentim_tta:all@adabncf_tta all@adabnrand_tta:all@adabncf_tta all@tent:all@adabncf \
  > logs/ablation5_alt.log 2>&1 || die "ablation5_alt summary"
$PY scripts/summarize_addons.py --prefix ablation3 --seeds 0 1 2 --overwrite \
  --tags all all_stain_default stain stain_he stain_soft stain_adv stain_uda \
  --modes plain tta adabncf adabncf_tta \
  --compare all@adabncf_tta:stain_uda@tta all@plain:stain_uda@plain all@adabncf_tta:stain_adv@adabncf_tta \
  > logs/ablation3.log 2>&1 || die "ablation3 summary"

echo "[$(ts)] PHASE2_DONE"
touch logs/phase2.done
