#!/bin/bash
#SBATCH --job-name=fmbench-demo2-lora
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
# Re-train the jun_model/demo2 Phikon-v2 LoRA mitosis classifiers on TGI, because the original adapters
# (trained on a Windows workstation) are not available. Phikon-v2 is public, so this needs no gated access;
# the Hibou-B and Virchow2 variants do.
# Methods, with the README's settings (40x patches, rank-8 LoRA on query/value, 8 epochs, patience 2,
# batch 8, source-scanner validation):
#   lora               README "Phikon-v2 LoRA Pilot"
#   lora_stain_grl001  + online HED stain augmentation (sigma 0.05, both) + scanner GRL 0.01, the best ablation
#   lora_stain, lora_grl001  the two single add-ons (not run by default)
# One output directory per method, jun_model/demo2/models/phikon_v2_40x_<method>_validation/, with
# <fold>_seed<k>_adapter.pt for every fold and seed, validation_runs.csv and summary.json. A method whose
# summary.json exists is skipped. Then writes configs/models/phikon_v2_<method>_demo2.yaml (aperio_cs2 fold,
# seed 0) for the benchmark.
# Submit: scripts/tgi/submit.sh demo2   (SEEDS="0" METHODS="lora" for less).
set -euo pipefail
source "$BENCH/scripts/tgi/env.sh"
REPO=$(cd "$BENCH/.." && pwd)
D=jun_model/demo2
SEEDS=${SEEDS:-0 1 2}
METHODS=${METHODS:-lora lora_stain_grl001}
echo "demo2 LoRA re-training on $(hostname), job ${SLURM_JOB_ID:-?}, methods [$METHODS], seeds [$SEEDS], $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash "$BENCH/scripts/tgi/link_breast_subset.sh"
cd "$REPO"
mkdir -p "$D/logs"
want=$(( $(wc -l < "$D/data/candidates.csv") - 1 ))
have=0  # (find on a missing directory fails, and with pipefail that would end the script)
[ -d "$D/patches/40x" ] && have=$(find "$D/patches/40x" -name '*.jpg' | wc -l)
if (( have < want )); then
  echo "extracting $want 40x candidate patches ($have present)"
  $PY "$D/extract_candidate_patches.py" --scales 40x > "$D/logs/extract_40x.log" 2>&1 || { tail -20 "$D/logs/extract_40x.log"; exit 1; }
  tail -2 "$D/logs/extract_40x.log"
fi

cache=$HF_HOME/hub   # shared with fmbench, so the benchmark restores the exact base revision
failed=0
for method in $METHODS; do
  case $method in
    lora)              label="LoRA";                    extra=() ;;
    lora_stain)        label="LoRA + stain";            extra=(--stain-augmentation --stain-sigma 0.05 --stain-mode both) ;;
    lora_grl001)       label="LoRA + GRL 0.01";         extra=(--grl-strength 0.01) ;;
    lora_stain_grl001) label="LoRA + stain + GRL 0.01"; extra=(--stain-augmentation --stain-sigma 0.05 --stain-mode both --grl-strength 0.01) ;;
    *) echo "unknown method $method" >&2; failed=1; continue ;;
  esac
  out=$D/models/phikon_v2_40x_${method}_validation
  if [ -f "$out/summary.json" ]; then echo "skip $method (done)"; else
    t0=$(date +%s)
    # shellcheck disable=SC2086
    if $PY -u "$D/run_lora_probe_validation.py" --model-key phikon_v2 --scale 40x --output-dir "$out" \
         --seeds $SEEDS --epochs 8 --patience 2 --batch-size 8 --eval-batch-size 16 \
         --cache-dir "$cache" --num-workers 12 "${extra[@]}" > "$D/logs/phikon_v2_40x_$method.log" 2>&1; then
      echo "$method done in $(( $(date +%s) - t0 ))s"
    else
      echo "FAILED $method"; tail -15 "$D/logs/phikon_v2_40x_$method.log"; failed=1; continue
    fi
  fi
  cfg=$BENCH/configs/models/phikon_v2_${method}_demo2.yaml
  if [ -f "$out/aperio_cs2_seed0_adapter.pt" ] && [ ! -f "$cfg" ]; then
    sed -e "s#^display_name: .*#display_name: Phikon-v2 + $label (demo2, MIDOG++ breast, CS2 fold, seed 0, re-trained on TGI)#" \
        -e "s#^  checkpoint: .*#  checkpoint: $out/aperio_cs2_seed0_adapter.pt\n  hf_cache_dir: \${HF_HOME}/hub#" \
        "$BENCH/configs/models/phikon_v2_lora_demo2.yaml.example" > "$cfg"
    echo "wrote $cfg"
  fi
done
echo "finished $(date -u +%FT%TZ)"
exit $failed
