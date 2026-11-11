#!/usr/bin/env bash
# Download and prepare every dataset that needs no form (about 150 GB of archives, roughly the
# same again extracted; see README.md). Safe to re-run: finished files are skipped and partial
# downloads resume.
#
#   FMBENCH_DATA=/data/fmbench scripts/download_all.sh
#   HEST=0 scripts/download_all.sh            # skip HEST-Benchmark (about 41 GB)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY -m fmbench download open           # pcam crc crc_nonorm wilds tils dlbcl midog
$PY -m fmbench prepare midog_nobreast dlbcl_he
if [ "${HEST:-1}" = "1" ]; then
  $PY -m fmbench hest-download
fi
$PY -m fmbench download mhist          # prints the form instructions (MHIST cannot be fetched by script)
$PY -m fmbench tasks --check
