#!/usr/bin/env bash
# Link the 150 annotated MIDOG++ human breast cancer TIFFs (jun_model/breast_cancer_subset/metadata) from
# the fmbench MIDOG++ download into jun_model/breast_cancer_subset/images/, the layout that
# jun_model/demo2 and aaron_model expect. That directory is git-ignored; the links are per file, so
# the 400-odd other MIDOG++ images stay out of it. Safe to run again.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
REPO=$(cd "$BENCH/.." && pwd)
sub=$REPO/jun_model/breast_cancer_subset
src=${FMBENCH_MIDOGPP:-$FMBENCH_DATA/midogpp}/images
mkdir -p "$sub/images"
n=0 missing=()
while read -r f; do
  if [ -s "$src/$f" ]; then ln -sfn "$src/$f" "$sub/images/$f"; n=$((n + 1)); else missing+=("$f"); fi
done < <(tail -n +2 "$sub/metadata/breast_cancer_slides.csv" | cut -d, -f2 | tr -d '\r')
echo "linked $n breast cancer images into $sub/images"
if (( ${#missing[@]} )); then
  echo "missing from $src: ${missing[*]:0:10}${missing[10]:+ ...} (${#missing[@]} files; run: python -m fmbench download midog)" >&2
  exit 1
fi
