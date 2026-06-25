#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

# Target unlabeled-size study.
# Values are total unlabeled target samples per target domain:
# 4000  -> 2000 fake + 2000 real
# 8000  -> 4000 fake + 4000 real
# 12000 -> 6000 fake + 6000 real

UNLABELED_SIZES=(4000 8000 12000)

for seed in 123 456; do
  for unlab_size in "${UNLABELED_SIZES[@]}"; do
    (
      T_LABELED_TOTAL=1000
      T_UNLABELED_TOTAL="$unlab_size"
      LAMBDA_MM=1.0
      LAMBDA_CA=0.1

      run_experiment "DF" "FS" "F2F" "$seed" "unlabeled_${unlab_size}"
    )
  done
done