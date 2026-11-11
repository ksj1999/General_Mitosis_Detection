#!/bin/bash
#SBATCH --job-name=fmbench-hest-rf
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
# HEST-Benchmark random forests (Virchow2 Table 3) for each model in $MODELS, from the embeddings that
# hest_embed.sbatch cached. No GPU: without cuML, scikit-learn fits the forests on this node's CPUs, one
# forest of 70 trees per gene (so at most 70 busy threads), 50 genes x 30 folds per embedding mode.
# Regressions already in results/runs.jsonl are skipped, so a resubmitted job picks up where it stopped.
# Submit with scripts/tgi/submit.sh hest-rf MODEL [MODEL ...], after the embedding job with
# DEPEND=afterany:<id> (afterok would never start if one model's embedding failed).
# Optional: HEST_MODES="cls", HEST_TASKS="SKCM HCC", RF_MAX_FEATURES=1.0 (the ablation; see NOTES.md).
set -euo pipefail
source "$BENCH/scripts/tgi/env.sh"
cd "$BENCH"
echo "fmbench HEST random forests [${MODELS:?}] on $(hostname), job ${SLURM_JOB_ID:-?}, $(nproc) CPUs, $(date -u +%FT%TZ)"
rc=0
for m in $MODELS; do  # a failing model must not stop the others
  echo "=== $m ($(date -u +%FT%TZ))"
  # shellcheck disable=SC2086
  $PY -m fmbench hest --model "$m" --stage regress --skip-done ${HEST_MODES:+--mode $HEST_MODES} \
    ${HEST_TASKS:+--hest-task $HEST_TASKS} ${RF_MAX_FEATURES:+--rf-max-features $RF_MAX_FEATURES} \
    || { echo "=== $m: FAILED (exit $?)"; rc=1; }
done
echo "finished $(date -u +%FT%TZ), exit $rc"
exit $rc
