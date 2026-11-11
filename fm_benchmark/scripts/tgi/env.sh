# Environment for fmbench on TGI (NCSA RAILS). Sourced by submit.sh and the *.sbatch jobs; also usable
# interactively on a login node:  source scripts/tgi/env.sh && $PY -m fmbench tasks --check
#
# Everything large lives in $HOME (/u/$USER: ~4.6 TB, 12M-file quota). /projects/bdbk is close to its
# 20M-file group quota, so nothing is written there; only the existing Hugging Face token is read from it.

BENCH=${BENCH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export BENCH
export FMBENCH_HOME=${FMBENCH_HOME:-$HOME/fmbench}
export FMBENCH_DATA=${FMBENCH_DATA:-$FMBENCH_HOME/data}      # raw downloads (~400 GB, ~1.1M files)
export FMBENCH_CACHE=${FMBENCH_CACHE:-$FMBENCH_HOME/cache}   # prepared tiles + embeddings
export FMBENCH_LOGS=${FMBENCH_LOGS:-$FMBENCH_HOME/logs}      # Slurm and download logs
# results stay in the checkout (fm_benchmark/results), so runs.jsonl and the leaderboard can be versioned

# Hugging Face: model weights and the HEST metadata cache in home; token from the usual HF_HOME.
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-${HF_HOME:-/projects/bdbk/$USER/.cache/huggingface}/token}
export HF_HOME=${FMBENCH_HF_HOME:-$FMBENCH_HOME/hf}
export HF_HUB_DISABLE_PROGRESS_BARS=1

export UV=${UV:-$HOME/.local/bin/uv}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}   # same filesystem as .venv, so uv can hardlink
export PY="$UV run --project $BENCH --no-sync python"
export PYTHONUNBUFFERED=1

mkdir -p "$FMBENCH_DATA" "$FMBENCH_CACHE" "$FMBENCH_LOGS" "$HF_HOME"
