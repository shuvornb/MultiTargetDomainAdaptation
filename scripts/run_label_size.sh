#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

# Target labeled-size study.
# Values are total labeled target samples per target domain:
# 0    -> 0 fake + 0 real
# 800  -> 400 fake + 400 real
# 1600 -> 800 fake + 800 real
# 2400 -> 1200 fake + 1200 real

LABEL_SIZES=(0 800 1600 2400)

for seed in 123 456; do
  for label_size in "${LABEL_SIZES[@]}"; do
    (
      T_LABELED_TOTAL="$label_size"
      T_UNLABELED_TOTAL=9000
      LAMBDA_MM=1.0
      LAMBDA_CA=0.1

      run_experiment "DF" "FS" "F2F" "$seed" "label_${label_size}"
    )
  done
done