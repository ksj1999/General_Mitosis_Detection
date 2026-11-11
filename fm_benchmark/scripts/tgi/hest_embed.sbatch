#!/bin/bash
#SBATCH --job-name=fmbench-hest-embed
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# Embed the HEST-Benchmark spots (228k patches, 10 tasks) for each model in $MODELS, one after another.
# The random forests run afterwards on a CPU node (hest_rf.sbatch), so this job holds the GPU only for
# the embedding. Submit with scripts/tgi/submit.sh hest-embed MODEL [MODEL ...]. Cached samples are skipped.
set -euo pipefail
source "$BENCH/scripts/tgi/env.sh"
cd "$BENCH"
echo "fmbench HEST embedding [${MODELS:?}] on $(hostname), job ${SLURM_JOB_ID:-?}, $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
rc=0
for m in $MODELS; do  # a failing model must not stop the others
  echo "=== $m ($(date -u +%FT%TZ))"
  $PY -m fmbench hest --model "$m" --stage embed --workers "${WORKERS:-14}" || { echo "=== $m: FAILED (exit $?)"; rc=1; }
done
echo "finished $(date -u +%FT%TZ), exit $rc"
exit $rc
