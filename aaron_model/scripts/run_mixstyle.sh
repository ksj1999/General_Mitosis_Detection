#!/bin/zsh
# MixStyle experiment: stain_mix first (main comparison vs stain), then mix (vs baseline); 3 seeds x 3 folds each.
cd "$(dirname "$0")/.."
for V in stain_mix mix; do
  for S in 0 1 2; do
    for F in CS2 S360 XR; do
      echo "[$(date +%H:%M:%S)] start $F $V seed$S"
      caffeinate -i .venv/bin/python scripts/train_patch.py --fold $F --variant $V --seed $S > logs/${F}_${V}_seed${S}.log 2>&1
      echo "[$(date +%H:%M:%S)] done  $F $V seed$S (exit $?)"
    done
  done
done
echo DONE > logs/mixstyle.done
