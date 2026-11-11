#!/usr/bin/env bash
# Submit fmbench jobs on TGI without going over the account-wide job cap.
#
#   scripts/tgi/submit.sh download                  # CPU: download + prepare all datasets, fetch + test every model
#   scripts/tgi/submit.sh smoke                     # GPU, 1 h: tests on CUDA, model check, 256-tile PCam run
#   scripts/tgi/submit.sh models virchow2 uni       # GPU: run_model.sh for these models, one after another
#   HEST=1 scripts/tgi/submit.sh models phikon      # ... and HEST-Benchmark too (random forests on the GPU node)
#   scripts/tgi/submit.sh hest-embed phikon uni     # GPU: embed the HEST-Benchmark spots only
#   DEPEND=afterany:<id> scripts/tgi/submit.sh hest-rf phikon uni  # CPU: HEST random forests from those embeddings
#   scripts/tgi/submit.sh aaron                     # GPU: re-train the aaron_model ResNet-18 ("all", 3 folds x 5 seeds)
#   scripts/tgi/submit.sh demo2                     # GPU: re-train the jun_model/demo2 Phikon-v2 LoRA classifiers
#   scripts/tgi/submit.sh retrain                   # GPU: aaron + demo2 in one job, then benchmark what they produced
#
# Settings (environment): ACCOUNT (default bdbk-tgirails: more fair share than bdbl, so A100 jobs start sooner), PARTITION (default gpu_a100 for GPU jobs),
# TIME (wall time), DEPEND (e.g. afterok:123456), MAX_JOBS (default 3).
#
# The cap counts running AND pending jobs (each array task separately) across every session of $USER:
# going above 3 jobs gets the account kicked off TGI. This script refuses to submit past it.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$BENCH"

MAX_JOBS=${MAX_JOBS:-3}
ACCOUNT=${ACCOUNT:-bdbk-tgirails}
kind=${1:?usage: $0 download|smoke|aaron|demo2|retrain|models|hest-embed|hest-rf MODEL [MODEL ...]}
shift || true

n=$(squeue -u "$USER" -r -h | wc -l)
if (( n + 1 > MAX_JOBS )); then
  echo "refusing to submit: $n job(s) already running or pending for $USER (cap $MAX_JOBS)" >&2
  squeue -u "$USER" -r -o "%.10i %.12P %.24j %.8T %.10M %.10l %R" >&2
  exit 1
fi

args=(--account="$ACCOUNT" --chdir="$BENCH" --export=ALL --output="$FMBENCH_LOGS/%x_%j.out")
[ -n "${TIME:-}" ] && args+=(--time="$TIME")
[ -n "${DEPEND:-}" ] && args+=(--dependency="$DEPEND")
case "$kind" in
  download) script=scripts/tgi/download.sbatch ;;
  aaron)    script=scripts/tgi/aaron_resnet18.sbatch ;;
  demo2)    script=scripts/tgi/demo2_lora.sbatch ;;
  retrain)  script=scripts/tgi/retrain.sbatch ;;
  smoke)    script=scripts/tgi/smoke.sbatch ;;
  models|hest-embed|hest-rf)
    (( $# )) || { echo "$kind: name at least one model config (python -m fmbench models)" >&2; exit 2; }
    export MODELS="$*"
    case "$kind" in
      models)     script=scripts/tgi/models.sbatch;     name="fmb-$1" ;;
      hest-embed) script=scripts/tgi/hest_embed.sbatch; name="fmb-hest-$1" ;;
      hest-rf)    script=scripts/tgi/hest_rf.sbatch;    name="fmb-hestrf-$1" ;;
    esac
    (( $# > 1 )) && name+="+$(( $# - 1 ))"  # a bare (( )) returning 1 inside $( ) would end the script under set -e
    args+=(--job-name="$name") ;;
  *) echo "unknown job kind: $kind" >&2; exit 2 ;;
esac
# GPU jobs default to gpu_a100. PARTITION=gpu (H100) is for training only, and never on rails05.
# The CPU jobs (download, hest-rf) keep the partition their sbatch file names.
if [ -n "${PARTITION:-}" ] && [ "$kind" != download ] && [ "$kind" != hest-rf ]; then
  args+=(--partition="$PARTITION")
  [ "$PARTITION" = gpu ] && args+=(--gres=gpu:h100:1 --exclude=rails05)
fi
sbatch "${args[@]}" "$script"
echo "jobs for $USER now: $(squeue -u "$USER" -r -h | wc -l) of $MAX_JOBS"
