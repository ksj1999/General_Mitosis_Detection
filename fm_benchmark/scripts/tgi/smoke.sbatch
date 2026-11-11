#!/bin/bash
#SBATCH --job-name=fmbench-smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Quick check of the GPU environment: unit tests, CUDA load of two open models, and (if PCam is ready)
# a 256-tile embed + 200-iteration probe. Smoke results go to results/smoke/, never the main leaderboard.
set -uo pipefail
source "$BENCH/scripts/tgi/env.sh"
cd "$BENCH"
echo "fmbench smoke on $(hostname), job ${SLURM_JOB_ID:-?}, $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch, timm, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), '| timm', timm.__version__, '| transformers', transformers.__version__)"
$PY -m pytest -q tests/ 2>&1 | tail -2
$PY -m fmbench models --check --device cuda resnet18_imagenet phikon
for m in resnet18_imagenet phikon; do
  $PY -m fmbench run --model "$m" --task pcam --limit 256 --iterations 200 --n-boot 50 --workers 8
done
echo "finished $(date -u +%FT%TZ)"
