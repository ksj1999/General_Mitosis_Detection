#!/bin/zsh
# Run all four variants for one fold (named by the held-out scanner).
# Usage: scripts/run_fold.sh CS2 [seed]
cd "$(dirname "$0")/.."
FOLD=$1; SEED=${2:-0}
for V in baseline stain adv stain_adv; do
  echo "[$(date +%H:%M:%S)] start $FOLD $V seed$SEED"
  caffeinate -i .venv/bin/python scripts/train_patch.py --fold $FOLD --variant $V --seed $SEED > logs/${FOLD}_${V}_seed${SEED}.log 2>&1
  echo "[$(date +%H:%M:%S)] done  $FOLD $V (exit $?)"
done
echo ALL_DONE
