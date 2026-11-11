#!/bin/bash
#SBATCH --job-name=fmbench-download
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
# Download, extract and prepare every dataset that needs no form, fetch HEST-Benchmark, and download and
# test-load every model config, all on a CPU node (compute nodes have internet). Submit with
# scripts/tgi/submit.sh download. Re-submitting resumes: finished files are skipped, partial ones resume.
set -uo pipefail
source "$BENCH/scripts/tgi/env.sh"
cd "$BENCH"
echo "fmbench download on $(hostname), job ${SLURM_JOB_ID:-?}, $(date -u +%FT%TZ)"
curl -sS -o /dev/null --max-time 30 https://zenodo.org || { echo "no internet on $(hostname)"; exit 2; }
$PY -m pytest -q tests/ 2>&1 | tail -2

# Tasks that share files (crc/crc_nonorm, dlbcl/dlbcl_he, midog/midog_nobreast) run in the same lane.
lane() {
  local name=$1; shift
  local log="$FMBENCH_LOGS/download_${SLURM_JOB_ID:-local}_${name}.log" rc=0
  for t in "$@"; do
    echo "== $t $(date -u +%FT%TZ)" >>"$log"
    if [ "$t" = hest ]; then cmd=(-m fmbench hest-download); else cmd=(-m fmbench download "$t"); fi
    if $PY "${cmd[@]}" >>"$log" 2>&1; then echo "done   $t $(date -u +%T)"; else echo "FAILED $t (see $log)"; rc=1; fi
  done
  return $rc
}
pids=()
lane pcam_crc pcam crc crc_nonorm & pids+=($!)
lane wilds_tils wilds tils & pids+=($!)
lane dlbcl dlbcl dlbcl_he & pids+=($!)
lane midog midog midog_nobreast & pids+=($!)
lane hest hest & pids+=($!)
# model weights go to $HF_HOME; each model is loaded on CPU and run on a dummy batch
models_log="$FMBENCH_LOGS/models_check_${SLURM_JOB_ID:-local}.log"
$PY -m fmbench models --check >"$models_log" 2>&1 &
models_pid=$!

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
wait "$models_pid"; echo "model check exit $? (see $models_log)"
grep -E '^(OK|FAILED|NOACCESS)' "$models_log"
$PY -m fmbench download mhist   # prints the form instructions; MHIST cannot be fetched by script
$PY -m fmbench tasks --check
du -sh "$FMBENCH_DATA" "$FMBENCH_CACHE" "$HF_HOME" 2>/dev/null
echo "finished $(date -u +%FT%TZ), status $status"
exit $status
