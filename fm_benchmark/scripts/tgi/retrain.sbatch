#!/bin/bash
#SBATCH --job-name=fmbench-retrain
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
# Both mitosis re-trainings in one job (so they take one of the three job slots), then the tile benchmark
# for every model they produced:
#   1. aaron_resnet18.sbatch  ResNet-18 "all" (3 folds x SEEDS_AARON, default 5 seeds)
#   2. demo2_lora.sbatch      Phikon-v2 LoRA, LoRA + stain + GRL (3 folds x SEEDS_DEMO2, default 3 seeds)
#   3. run_model.sh           resnet18_midog_all, phikon_v2_lora_demo2, phikon_v2_lora_stain_grl001_demo2
# Each step skips work that is already done, so the job can be submitted again after a failure.
# Submit: scripts/tgi/submit.sh retrain   (BENCHMARK=0 to stop after training; BENCH_MODELS="..." to pick).
set -uo pipefail
source "$BENCH/scripts/tgi/env.sh"
cd "$BENCH"
echo "fmbench retrain on $(hostname), job ${SLURM_JOB_ID:-?}, $(date -u +%FT%TZ)"
rc=0
SEEDS=${SEEDS_AARON:-0 1 2 3 4} bash scripts/tgi/aaron_resnet18.sbatch || rc=1
SEEDS=${SEEDS_DEMO2:-0 1 2} bash scripts/tgi/demo2_lora.sbatch || rc=1

if [ "${BENCHMARK:-1}" = "1" ]; then
  models=()
  # BENCH_MODELS limits the benchmark to some of them, e.g. when a re-submission only adds the LoRA models
  for m in ${BENCH_MODELS:-resnet18_midog_all phikon_v2_lora_demo2 phikon_v2_lora_stain_grl001_demo2}; do
    [ -f "configs/models/$m.yaml" ] && models+=("$m")
  done
  if (( ${#models[@]} )); then
    WORKERS=${WORKERS:-14} PYTHON="$PY" scripts/run_model.sh "${models[@]}" || rc=1
  else
    echo "no re-trained model configs to benchmark"; rc=1
  fi
fi
echo "finished $(date -u +%FT%TZ), exit $rc"
exit $rc
