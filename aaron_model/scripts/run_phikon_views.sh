#!/bin/zsh
# After the 128-px Phikon run finishes, probe Phikon on the two 224-px views (plain features only).
cd "$(dirname "$0")/.."
while pgrep -f "phikon_probe.py$" > /dev/null || pgrep -f "phikon_probe.py " > /dev/null; do sleep 10; done
while [ ! -f data/patches_224_um112.npy ]; do sleep 5; done
for V in um056 um112; do
  echo "[$(date +%H:%M:%S)] start phikon_$V"
  caffeinate -i .venv/bin/python scripts/phikon_probe.py --patches data/patches_224_$V.npy --name phikon_$V --n-aug 0
  echo "[$(date +%H:%M:%S)] done  phikon_$V (exit $?)"
done
echo DONE > logs/phikon_views.done
