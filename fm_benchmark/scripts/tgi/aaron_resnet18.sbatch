#!/bin/bash
#SBATCH --job-name=fmbench-aaron-r18
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Re-train the aaron_model ResNet-18 "all" configuration on TGI, because the original checkpoints (trained
# on a Mac) are not available. "all" = --variant stain_adv --stain-mode he_only --soft-labels
# (aaron_model/jobs/phase2.txt, ADDONS_PROTOCOL.md), one model per held-out scanner (CS2, S360, XR) and seed.
# Input: the 150 annotated MIDOG++ human breast cancer TIFFs that `fmbench download midog` fetched, plus
# jun_model/breast_cancer_subset/metadata. Output: aaron_model/{data,checkpoints,results,logs}/ in this
# checkout, and configs/models/resnet18_midog_all.yaml for the benchmark (CS2 fold, seed 0).
# Submit: scripts/tgi/submit.sh aaron   (SEEDS="0" for a single seed). Finished runs are skipped.
set -euo pipefail
source "$BENCH/scripts/tgi/env.sh"
REPO=$(cd "$BENCH/.." && pwd)
A=$REPO/aaron_model
SEEDS=${SEEDS:-0 1 2 3 4}
echo "aaron_model re-training on $(hostname), job ${SLURM_JOB_ID:-?}, seeds $SEEDS, $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# the layout scripts/prepare_patches.py expects: <dir>/images/*.tiff and <dir>/metadata/
bash "$BENCH/scripts/tgi/link_breast_subset.sh"
src=$REPO/jun_model/breast_cancer_subset

cd "$A"
mkdir -p data logs results checkpoints
if [ ! -f data/patches_128.npz ]; then
  $PY scripts/prepare_patches.py --data-dir "$src" > logs/prepare_patches.log 2>&1 || { tail -20 logs/prepare_patches.log; exit 1; }
  tail -12 logs/prepare_patches.log
fi

failed=0
for seed in $SEEDS; do
  for fold in CS2 S360 XR; do
    name=${fold}_all_seed${seed}
    if [ -f "results/$name.json" ]; then echo "skip $name (done)"; continue; fi
    t0=$(date +%s)
    if $PY scripts/train_patch.py --fold "$fold" --seed "$seed" --tag all \
         --variant stain_adv --stain-mode he_only --soft-labels > "logs/$name.log" 2>&1; then
      echo "$name $(( $(date +%s) - t0 ))s $(tail -1 "logs/$name.log")"
    else
      echo "FAILED $name"; tail -5 "logs/$name.log"; failed=1
    fi
  done
done

cfg=$BENCH/configs/models/resnet18_midog_all.yaml
if [ -f "$A/checkpoints/CS2_all_seed0.pt" ] && [ ! -f "$cfg" ]; then
  sed 's/^display_name: .*/display_name: ResNet-18 MIDOG++ breast (all, CS2 fold, seed 0, re-trained on TGI)/' \
    "$cfg.example" > "$cfg"
  echo "wrote $cfg"
fi
echo "finished $(date -u +%FT%TZ)"
exit $failed
