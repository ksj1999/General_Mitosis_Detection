#!/bin/bash
#SBATCH --job-name=fmbench-models
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
# Embed + linear-probe every task for each model in $MODELS, one after another (scripts/run_model.sh).
# Submit with scripts/tgi/submit.sh models MODEL [MODEL ...]. Optional: SEEDS="0 1 2", TASKS="pcam crc",
# HEST=1. Embeddings are cached, so a re-submitted job only redoes what is missing.
# One A100 costs 48 SU/h for up to 64 CPUs / 128 GB, so 16 CPUs / 96 GB cost nothing extra.
set -euo pipefail
source "$BENCH/scripts/tgi/env.sh"
cd "$BENCH"
echo "fmbench models [${MODELS:?}] on $(hostname), job ${SLURM_JOB_ID:-?}, $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
WORKERS=${WORKERS:-14} PYTHON="$PY" scripts/run_model.sh $MODELS
echo "finished $(date -u +%FT%TZ)"
