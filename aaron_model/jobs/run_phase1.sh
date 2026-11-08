#!/bin/zsh
# Phase 1 of the add-on study (ADDONS_PROTOCOL.md), strictly serial on the GPU
# (concurrent MPS processes produced a Metal command-buffer error on 2026-11-07).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NEW="stain_s002 stain_s010 stain_he stain_scale02 stain_ema stain_soft macenko macenko_stain"
ts() { date +%H:%M:%S; }

echo "[$(ts)] verify TENT outputs written near the Metal error (S360 baseline/stain/adv)"
$PY scripts/eval_tent.py --tags baseline stain adv --folds S360 --seeds 0 1 2 --out-prefix verify_

echo "[$(ts)] training queue"
caffeinate -i $PY scripts/run_queue.py jobs/phase1.txt --workers 1

echo "[$(ts)] TTA / AdaBN on new tags"
$PY scripts/eval_tta_adabn.py --tags ${=NEW} --seeds 0 1 2
echo "[$(ts)] cross-fitted AdaBN on new tags"
$PY scripts/eval_adabn_crossfit.py --tags ${=NEW} --seeds 0 1 2
echo "[$(ts)] UDA merge"
$PY scripts/eval_uda.py --base stain_uda --seeds 0 1 2

echo "[$(ts)] PHASE1_DONE"
touch logs/phase1.done
