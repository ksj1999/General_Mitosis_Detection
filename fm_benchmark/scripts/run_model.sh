#!/usr/bin/env bash
# Run the Virchow2 tile benchmark for one or more models on a GPU machine.
#
#   scripts/run_model.sh virchow2 resnet18_imagenet
#   TASKS="pcam crc" SEEDS="0 1 2" scripts/run_model.sh virchow2
#   HEST=1 scripts/run_model.sh virchow2       # also HEST-Benchmark (random forest; needs cuML to be fast)
#
# Embeddings are cached, so a second run only trains the probes. Results are appended to
# results/runs.jsonl and the leaderboard is rebuilt at the end. Tasks whose data is missing are
# skipped with a message.
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS=${TASKS:-paper8 extra}
SEEDS=${SEEDS:-0}
WORKERS=${WORKERS:-8}
PY=${PYTHON:-python}

if [ "$#" -eq 0 ]; then
  echo "usage: $0 MODEL [MODEL ...]   (see: $PY -m fmbench models)" >&2
  exit 2
fi

rc=0
for model in "$@"; do  # a failing model must not stop the others
  echo "=== $model ($(date -u +%FT%TZ))"
  # shellcheck disable=SC2086
  $PY -m fmbench run --model "$model" --task $TASKS --seeds $SEEDS --workers "$WORKERS" || { echo "=== $model: FAILED (exit $?)"; rc=1; }
  if [ "${HEST:-0}" = "1" ]; then
    $PY -m fmbench hest --model "$model" --workers "$WORKERS" || { echo "=== $model HEST: FAILED (exit $?)"; rc=1; }
  fi
done
$PY -m fmbench leaderboard || rc=1
exit $rc
